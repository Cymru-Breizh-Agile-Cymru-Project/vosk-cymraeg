from functools import reduce
from pathlib import Path

import datasets
import polars as pl

from vosk_cymraeg.datasets.hf_utils import dump_dataset_audio_files


def fetch_lleisiau_arfor(output_path: Path) -> None:
    # Since we are processing individual clips we need to
    speaker_count = 0

    dataset_splits = ["train_clean", "dev_clean", "test_clean"]
    for split in dataset_splits:
        ds: datasets.Dataset = datasets.load_dataset(
            "cymen-arfor/lleisiau-arfor", split=split
        )
        len_before = len(ds)
        ds = ds.filter(lambda lang: lang == "cy", input_columns=["language"])
        print(
            f"Filtered {len_before - len(ds)} entries ({(len_before - len(ds)) / len_before:.2%})"
        )

        # Same as Banc: Since we don't have any info every utterance is a unique speaker
        ds = ds.add_column(
            "speaker", [f"lla-{i + speaker_count:04d}" for i in range(len(ds))]
        )
        ds = ds.add_column(
            "utterance", [f"lla-{i + speaker_count:04d}-0000" for i in range(len(ds))]
        )
        speaker_count += len(ds)

        dump_dataset_audio_files(ds, output_path)

        # Audio can then be dumped to save memory
        ds = ds.remove_columns("audio")
        ds.to_csv(
            output_path / f"{split}.csv", columns=["speaker", "utterance", "sentence"]
        )

    # Combine all datasets into one
    dfs = [pl.read_csv(output_path / f"{split}.csv") for split in dataset_splits]
    all_df = pl.concat(dfs)
    total_length = reduce(lambda acc, df: acc + len(df), dfs, 0)

    # Assert that there aren't any duplicate utterances
    assert len(all_df) == total_length
    assert all_df["utterance"].is_unique().all()
    all_df.write_csv(output_path / "all.csv")
