"""Скачивание выбранных сабсетов VK-LSVD в data/raw/VK-LSVD.

Повторный запуск докачивает, а не качает заново.
item_embeddings.npz (~2.5 ГБ) качается только с --embeddings.
"""
import argparse

from src.data import load_in_memory, week_files

SUBSETS = {
    'dense': 'up0.001_ip0.001',
    'sparse-a': 'ur0.01_ip0.01',
    'sparse-b': 'ur0.01_ir0.01',
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('subsets', nargs='+', choices=sorted(SUBSETS))
    parser.add_argument('--train-weeks', type=int, default=25)
    parser.add_argument('--embeddings', action='store_true')
    args = parser.parse_args()

    metadata = ['metadata/users_metadata.parquet', 'metadata/items_metadata.parquet']
    if args.embeddings:
        metadata.append('metadata/item_embeddings.npz')

    for i, key in enumerate(args.subsets):
        subset = SUBSETS[key]
        print(f'=== {key} -> {subset} ===', flush=True)
        load_in_memory(
            week_files(subset, range(args.train_weeks), 'train'),
            week_files(subset, [25], 'validation'),
            metadata if i == 0 else None,
        )
    print('done', flush=True)


if __name__ == '__main__':
    main()
