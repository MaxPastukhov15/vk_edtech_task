"""Аудит состава позитивов VK-LSVD.

Главный вопрос: какая доля позитивов держится только на click_on_author /
open_comments — без лайка, шера, закладки и досмотра.

Определения (docs/problem_statement.md §3), на строке после исключения dislike:

    watch(t)      = duration > 0 и timespent / duration >= t
    strong        = like | share | bookmark
    disputed      = click_on_author | open_comments
    positive      = (strong | disputed | watch(0.5)) & ~dislike
    disputed_only = disputed & ~strong & ~watch(0.5) & ~dislike

p-core и дедуп не применяются: p-core сам считается по позитивам, то есть зависит
от определения, которое мы проверяем. Знаменатель — сырые позитивы.
"""
import argparse
import sys
from pathlib import Path

import polars as pl

from src.data import RAW_ROOT, scan_subset

SUBSETS = {
    'dense': 'up0.001_ip0.001',
    'sparse-a': 'ur0.01_ip0.01',
    'sparse-b': 'ur0.01_ir0.01',
}

FLAGS = ['like', 'dislike', 'share', 'bookmark', 'click_on_author', 'open_comments']
COLUMNS = ['user_id', 'item_id', 'timespent'] + FLAGS

BASE_THRESHOLD = 0.5
LOOSE_THRESHOLD = 0.3


def watch(threshold: float) -> pl.Expr:
    ratio = pl.col('timespent').cast(pl.Float64) / pl.col('duration').cast(pl.Float64)
    return ((pl.col('duration') > 0) & (ratio >= threshold)).fill_null(False)


STRONG = pl.col('like') | pl.col('share') | pl.col('bookmark')
DISPUTED = pl.col('click_on_author') | pl.col('open_comments')
NOT_DISLIKE = ~pl.col('dislike')


def positive(threshold: float | None) -> pl.Expr:
    """threshold=None -> strict: только явный фидбэк, без timespent."""
    explicit = STRONG | DISPUTED
    signal = explicit if threshold is None else explicit | watch(threshold)
    return signal & NOT_DISLIKE


def disputed_only(threshold: float | None) -> pl.Expr:
    rest = STRONG if threshold is None else STRONG | watch(threshold)
    return DISPUTED & ~rest & NOT_DISLIKE


def with_duration(lazy: pl.LazyFrame, root: str) -> pl.LazyFrame:
    items = (pl.scan_parquet(f'{root}/metadata/items_metadata.parquet')
             .select('item_id', 'duration'))
    return lazy.join(items, on='item_id', how='left')


def counts(lazy: pl.LazyFrame) -> dict[str, int]:
    """Один стриминговый проход: все счётчики сразу."""
    aggregates = {f'n_{flag}': pl.col(flag).sum() for flag in FLAGS}
    aggregates.update({
        'n_rows': pl.len(),
        'n_watch_050': watch(BASE_THRESHOLD).sum(),
        'n_watch_030': watch(LOOSE_THRESHOLD).sum(),
        'pos_strict': positive(None).sum(),
        'pos_base': positive(BASE_THRESHOLD).sum(),
        'pos_loose': positive(LOOSE_THRESHOLD).sum(),
        'disputed_only_strict': disputed_only(None).sum(),
        'disputed_only_base': disputed_only(BASE_THRESHOLD).sum(),
        'disputed_only_loose': disputed_only(LOOSE_THRESHOLD).sum(),
        'watch_only_base': (watch(BASE_THRESHOLD) & ~STRONG & ~DISPUTED & NOT_DISLIKE).sum(),
        'strong_only_base': (STRONG & ~DISPUTED & ~watch(BASE_THRESHOLD) & NOT_DISLIKE).sum(),
        # независимая проверка: pos_base - |(strong|watch) & ~dislike| == disputed_only_base
        'check_without_disputed': ((STRONG | watch(BASE_THRESHOLD)) & NOT_DISLIKE).sum(),
        'n_duration_zero': (pl.col('duration') == 0).sum(),
        'n_duration_null': pl.col('duration').is_null().sum(),
    })
    null_flags = pl.lit(0, dtype=pl.Int64)
    for flag in FLAGS:
        null_flags = null_flags + pl.col(flag).is_null().cast(pl.Int64)
    aggregates['n_flag_null'] = null_flags.sum()

    row = lazy.select(**aggregates).collect(engine='streaming')
    return {name: int(row[name][0]) for name in row.columns}


def last_target_stats(lazy: pl.LazyFrame, root: str) -> dict[str, int]:
    """GTS Last: последний позитив пользователя в неделе валидации.

    Строки внутри недельного файла упорядочены по времени, поэтому порядковый
    номер строки — это порядок во времени.

    Два места, где порядок легко потерять, и оба дают разный ответ от прогона
    к прогону, а не падение:
    1. with_row_index идёт ДО джойна с duration — join в polars по умолчанию
       не сохраняет порядок строк (maintain_order='none');
    2. sort_by стоит ВНУТРИ agg — group_by не гарантирует порядок внутри группы,
       поэтому last() после внешнего sort ненадёжен.
    """
    last_positive = (
        with_duration(lazy.with_row_index('row_nr'), root)
        .filter(positive(BASE_THRESHOLD))
        .select('user_id', 'row_nr', target_disputed=disputed_only(BASE_THRESHOLD))
        .group_by('user_id')
        .agg(pl.col('target_disputed').sort_by('row_nr').last())
    )
    row = last_positive.select(
        users_with_target=pl.len(),
        targets_disputed=pl.col('target_disputed').sum(),
    ).collect(engine='streaming')
    return {name: int(row[name][0]) for name in row.columns}


def markdown_table(rows: list[dict[str, object]]) -> str:
    header = list(rows[0])
    lines = ['| ' + ' | '.join(header) + ' |',
             '| ' + ' | '.join('---' for _ in header) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row[key]) for key in header) + ' |')
    return '\n'.join(lines)


def thousands(value: int) -> str:
    return f'{value:,}'.replace(',', ' ')


def share(part: int, whole: int) -> str:
    return '-' if whole == 0 else f'{100 * part / whole:.3f}%'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('subsets', nargs='+', choices=sorted(SUBSETS))
    parser.add_argument('--train-weeks', type=int, default=25)
    parser.add_argument('--root', default=RAW_ROOT)
    parser.add_argument('--out', default='experiments/positive_mix.md')
    parser.add_argument('--samples', action='store_true')
    args = parser.parse_args()

    summary: list[dict[str, object]] = []
    detail: list[dict[str, object]] = []
    failures: list[str] = []

    for key in args.subsets:
        subset = SUBSETS[key]
        train, validation = scan_subset(subset, range(args.train_weeks), [25],
                                        COLUMNS, args.root)
        splits = {'train (week_00-24)': train, 'validation (week_25)': validation}

        for split_name, lazy in splits.items():
            stat = counts(with_duration(lazy, args.root))

            identity = stat['pos_base'] - stat['check_without_disputed']
            if identity != stat['disputed_only_base']:
                failures.append(
                    f'{key}/{split_name}: тождество нарушено, '
                    f'{identity} != {stat["disputed_only_base"]}')
            if not stat['pos_strict'] <= stat['pos_base'] <= stat['pos_loose']:
                failures.append(f'{key}/{split_name}: strict <= base <= loose нарушено')
            if stat['disputed_only_loose'] > stat['disputed_only_base']:
                failures.append(f'{key}/{split_name}: disputed_only не монотонен по порогу')
            if stat['n_flag_null']:
                failures.append(f'{key}/{split_name}: {stat["n_flag_null"]} null в флагах')

            summary.append({
                'сабсет': key,
                'split': split_name,
                'строк': thousands(stat['n_rows']),
                'позитивов (base)': thousands(stat['pos_base']),
                'спорных only': thousands(stat['disputed_only_base']),
                'доля спорных': share(stat['disputed_only_base'], stat['pos_base']),
                'она же при loose 0.3': share(stat['disputed_only_loose'], stat['pos_loose']),
                'она же при strict': share(stat['disputed_only_strict'], stat['pos_strict']),
                'только досмотр': share(stat['watch_only_base'], stat['pos_base']),
            })
            detail.append(dict({'сабсет': key, 'split': split_name},
                               **{name: thousands(value) for name, value in stat.items()}))

        targets = last_target_stats(validation, args.root)
        summary.append({
            'сабсет': key,
            'split': 'таргеты GTS Last (week_25)',
            'строк': '-',
            'позитивов (base)': thousands(targets['users_with_target']),
            'спорных only': thousands(targets['targets_disputed']),
            'доля спорных': share(targets['targets_disputed'], targets['users_with_target']),
            'она же при loose 0.3': '-',
            'она же при strict': '-',
            'только досмотр': '-',
        })

    report = '\n'.join([
        '# Состав позитивов VK-LSVD',
        '',
        'Сгенерировано `scripts/positive_mix_audit.py`. Определения — '
        '`docs/problem_statement.md` §3; p-core и дедуп не применялись.',
        '',
        '## Главное',
        '',
        markdown_table(summary),
        '',
        'В строке «таргеты GTS Last» колонка «позитивов» — число пользователей '
        'с хотя бы одним позитивом в неделе валидации, «спорных only» — у скольких '
        'из них последний позитив недели держится только на спорных флагах.',
        '',
        '## Все счётчики',
        '',
        markdown_table(detail),
        '',
    ])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding='utf-8')
    print(report)

    if args.samples:
        for key in args.subsets:
            _, validation = scan_subset(SUBSETS[key], range(args.train_weeks), [25],
                                        COLUMNS, args.root)
            validation = with_duration(validation, args.root)
            samples = [('disputed_only', disputed_only(BASE_THRESHOLD)),
                       ('watch_only', watch(BASE_THRESHOLD) & ~STRONG & ~DISPUTED & NOT_DISLIKE)]
            for label, condition in samples:
                print(f'\n--- {key}: {label} ---')
                print(validation.filter(condition).head(5).collect(engine='streaming'))

    if failures:
        print('\nПРОВЕРКИ НЕ ПРОШЛИ:', file=sys.stderr)
        for failure in failures:
            print(' -', failure, file=sys.stderr)
        raise SystemExit(1)
    print('\nвсе проверки целостности пройдены')


if __name__ == '__main__':
    main()
