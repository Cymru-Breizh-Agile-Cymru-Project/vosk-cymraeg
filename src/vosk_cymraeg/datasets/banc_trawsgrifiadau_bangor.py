import logging
import os
from pathlib import Path

import datasets
from dotenv import load_dotenv

from vosk_cymraeg.datasets.hf_utils import (
    create_combined_split,
    dump_dataset_audio_files,
)


def fetch_banc_trawsgrifiadau_bangor(output_path: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info(
        "Loading dataset 'techiaith/banc-trawsgrifiadau-bangor' from HuggingFace"
    )

    load_dotenv()
    token = os.environ["HF_TOKEN"]
    assert token

    speaker_count = 0
    dataset_splits = ["train", "validation", "test"]

    for split in dataset_splits:
        ds: datasets.Dataset = datasets.load_dataset(
            "techiaith/banc-trawsgrifiadau-bangor", split=split, token=token
        )

        # Since the data contains no speaker information we can treat all utterances as unique
        # speakers which means that the format for each clip is btb-<speaker>-0000
        ds = ds.add_column(
            "speaker", [f"btb-{i + speaker_count:04d}" for i in range(len(ds))]
        )
        ds = ds.add_column(
            "utterance", [f"btb-{i + speaker_count:04d}-0000" for i in range(len(ds))]
        )

        # Update speaker count
        speaker_count += len(ds)

        # Batch dumps all of the bytes in audio
        ds = ds.add_column("path", dump_dataset_audio_files(ds, output_path))
        ds = ds.add_column("lang", ["cy" for _ in range(len(ds))])

        # Audio can then be dumped to save memory
        ds = ds.remove_columns("audio")
        ds.to_csv(
            output_path / f"{split}.csv",
            columns=["speaker", "utterance", "path", "lang", "sentence"],
        )

    # Combine all datasets into one
    all_df = create_combined_split(output_path, dataset_splits)
    all_df.write_csv(output_path / "all.csv")
