from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download

HF_REPO = 'deepvk/VK-LSVD'
RAW_ROOT = 'data/raw/VK-LSVD'
PROCESSED_ROOT = 'data/processed'


def week_files(subsample_name: str, weeks: Iterable[int] = range(25),
               split: str = 'train', root: Optional[str] = None) -> list[str]:
    """Пути недельных файлов сабсета. root=None -> пути относительно репозитория HF."""
    prefix = f'{root}/' if root else ''
    return [f'{prefix}subsamples/{subsample_name}/{split}/week_{week:02}.parquet'
            for week in weeks]


def scan_subset(subsample_name: str = 'up0.001_ip0.001',
                train_weeks: Iterable[int] = range(25),
                val_weeks: Iterable[int] = (25,),
                columns: Optional[Sequence[str]] = None,
                root: str = RAW_ROOT) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Ленивая склейка недель. columns задаёт проекцию, чтобы не читать лишние колонки."""
    def scan(split: str, weeks: Iterable[int]) -> pl.LazyFrame:
        frames = [pl.scan_parquet(file)
                  for file in week_files(subsample_name, weeks, split, root)]
        lazy = pl.concat(frames) if len(frames) > 1 else frames[0]
        return lazy.select(columns) if columns else lazy

    return scan('train', train_weeks), scan('validation', val_weeks)


def load_subset(subsample_name: str = 'up0.001_ip0.001',
                train_weeks: Iterable[int] = range(25),
                val_weeks: Iterable[int] = (25,),
                columns: Optional[Sequence[str]] = None,
                root: str = RAW_ROOT) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_interactions, val_interactions = scan_subset(
        subsample_name, train_weeks, val_weeks, columns, root)
    return (train_interactions.collect(engine='streaming'),
            val_interactions.collect(engine='streaming'))


def load_metadata(root: str = RAW_ROOT,
                  file: Optional[dict[str, str]] = None) -> dict[str, pl.DataFrame]:
    file = file or {'users': f'{root}/metadata/users_metadata.parquet',
                    'items': f'{root}/metadata/items_metadata.parquet'}
    return {'users': pl.read_parquet(file['users']),
            'items': pl.read_parquet(file['items'])}


def load_embeddings(train_items: pl.DataFrame,
                    file: str = f'{RAW_ROOT}/metadata/item_embeddings.npz',
                    dim: int = 64) -> pl.DataFrame:
    archive = np.load(file)
    item_ids = archive['item_id']
    item_embeddings = archive['embedding']

    mask = np.isin(item_ids, train_items.to_numpy())
    item_ids = item_ids[mask]
    item_embeddings = item_embeddings[mask][:, :dim]
    return pl.DataFrame({'item_id': item_ids, 'embedding': item_embeddings})


def load_in_memory(train_interactions_files: list[str],
                   val_interactions_file: list[str],
                   metadata_files: Optional[list[str]] = None,
                   local_dir: str = RAW_ROOT) -> None:
    files: list[str] = train_interactions_files + val_interactions_file
    if metadata_files:
        files += metadata_files

    for file in files:
        hf_hub_download(
            repo_id=HF_REPO, repo_type='dataset',
            filename=file, local_dir=local_dir
        )


def save_processed_metadata(files: list[str], metadata: list[pl.DataFrame]) -> None:
    if len(files) != len(metadata):
        raise ValueError("number of filenames and number of metadata must be equal")

    for file, meta in zip(files, metadata):
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        meta.write_parquet(file)


if __name__ == "__main__":
    subsample_name = 'up0.001_ip0.001'
    content_embedding_size = 32

    load_in_memory(
        week_files(subsample_name, range(25), 'train'),
        week_files(subsample_name, [25], 'validation'),
        ['metadata/users_metadata.parquet',
         'metadata/items_metadata.parquet',
         'metadata/item_embeddings.npz'],
    )

    train_interactions, val_interactions = load_subset(subsample_name)

    train_users = train_interactions.select('user_id').unique()
    train_items = train_interactions.select('item_id').unique()

    embeddings = load_embeddings(train_items, dim=content_embedding_size)

    users_metadata, items_metadata = load_metadata().values()

    users_metadata = users_metadata.join(train_users, on='user_id')
    items_metadata = items_metadata.join(train_items, on='item_id')
    items_metadata = items_metadata.join(embeddings, on='item_id')

    save_processed_metadata(
        [f'{PROCESSED_ROOT}/{subsample_name}/{name}.parquet'
         for name in ['users_metadata', 'items_metadata']],
        [users_metadata, items_metadata],
    )
