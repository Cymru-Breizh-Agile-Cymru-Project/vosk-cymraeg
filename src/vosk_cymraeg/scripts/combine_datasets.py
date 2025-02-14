from pathlib import Path

import polars as pl


def main():
    """Combines all the predefined splits for the various datasets into a single split
    to create a combined dataset"""
    output_path = Path("data/processed/dataset")
    output_path.mkdir(parents=True, exist_ok=True)
    interim_path = Path("data/interim")

    print(f"Exporting combined dataset to {output_path}")

    def combine_and_write(name: str, files: list[str]) -> None:
        """Reads the files file polars (paths given relative to the interim folder) and
        exports them to the output path with the given name"""
        pl.concat([pl.read_csv(interim_path / file) for file in files]).write_csv(
            output_path / f"{name}.csv"
        )

    for dataset in ["banc", "cy/cy", "lleisiau_arfor"]:
        src_clips = interim_path / Path(dataset).parent / "clips"
        target_clips = output_path / "clips"
        target_clips.mkdir(exist_ok=True)
        for wav_file in src_clips.glob("*.wav"):
            target_wav_file = target_clips / wav_file.name
            if not target_wav_file.exists(): 
                target_wav_file.symlink_to(wav_file.absolute())


    # Training set
    combine_and_write(
        "train", ["banc/train.csv", "cv/cy/train.csv", "lleisiau_arfor/train_clean.csv"]
    )
    # Validation set
    combine_and_write(
        "dev", ["banc/validation.csv", "cv/cy/dev.csv", "lleisiau_arfor/dev_clean.csv"]
    )
    # Test set
    combine_and_write(
        "test", ["banc/test.csv", "cv/cy/test.csv", "lleisiau_arfor/test_clean.csv"]
    )
