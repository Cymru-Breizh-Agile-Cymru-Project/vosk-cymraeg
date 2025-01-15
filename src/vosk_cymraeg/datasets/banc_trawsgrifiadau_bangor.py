from pathlib import Path

import datasets

from vosk_cymraeg.datasets.hf_utils import dump_dataset_audio_files


def fetch_banc_trawsgrifiadau_bangor(output_path: Path) -> None:
    print("Fetching techiaith/banc-trawsgrifiadau-bangor from HuggingFace")
    all_ds: datasets.Dataset = datasets.load_dataset(
        "techiaith/banc-trawsgrifiadau-bangor", split="clips"
    )

    # Since the data contains no speaker information we can treat all utterances as unique
    # speakers which means that the format for each clip is btb-<speaker>-0000
    print("Generating speaker and utterance columns")
    all_ds = all_ds.add_column("speaker", [f"btb-{i:04d}" for i in range(len(all_ds))])
    all_ds = all_ds.add_column(
        "utterance", [f"btb-{i:04d}-0000" for i in range(len(all_ds))]
    )

    # Batch dumps all of the bytes in audio
    dump_dataset_audio_files(all_ds, output_path)

    # Audio can then be dumped to save memory
    all_ds = all_ds.remove_columns("audio")
    all_ds.to_csv(output_path / "all.csv", columns=["speaker", "utterance", "sentence"])
