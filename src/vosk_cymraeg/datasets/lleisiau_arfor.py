import logging
from pathlib import Path

import datasets

from vosk_cymraeg.datasets.hf_utils import (
    create_combined_split,
    dump_dataset_audio_files,
)


def fetch_lleisiau_arfor(output_path: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Loading dataset 'cymen-arfor/lleisiau-arfor' from HuggingFace")

    # Since we are processing individual clips we need to
    speaker_count = 0

    dataset_splits = ["train_clean", "dev_clean", "test_clean"]
    for split in dataset_splits:
        ds: datasets.Dataset = datasets.load_dataset(
            "cymen-arfor/lleisiau-arfor", split=split
        )
        ds = ds.rename_column("language", "lang")

        # Same as Banc: Since we don't have any info every utterance is a unique speaker
        ds = ds.add_column(
            "speaker", [f"lla-{i + speaker_count:04d}" for i in range(len(ds))]
        )
        ds = ds.add_column(
            "utterance", [f"lla-{i + speaker_count:04d}-0000" for i in range(len(ds))]
        )
        speaker_count += len(ds)

        ds = ds.add_column("path", dump_dataset_audio_files(ds, output_path))

        # Audio can then be dumped to save memory
        ds = ds.remove_columns("audio")
        ds.to_csv(
            output_path / f"{split}.csv",
            columns=["speaker", "utterance", "path", "lang", "sentence"],
        )

    # Combine all datasets into one

    # Combine all datasets into one
    all_df = create_combined_split(output_path, dataset_splits)
    all_df.write_csv(output_path / "all.csv")
