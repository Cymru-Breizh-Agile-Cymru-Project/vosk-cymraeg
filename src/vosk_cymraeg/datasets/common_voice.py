from pathlib import Path

import polars as pl
import sox
from tqdm import tqdm

DATASET_SPLITS = ["train", "test", "dev", "other"]


def process_common_voice(input_path: Path, output_path: Path) -> None:
    # Determine length of speaker IDs
    cid_length = determine_cid_length(input_path)
    print(f"Smallest N that still yields unique client IDs is {cid_length}")

    for split in tqdm(DATASET_SPLITS, desc="Converting splits"):
        # Load the data and construct required columns
        df = (
            pl.scan_csv(input_path / f"{split}.tsv", separator="\t", quote_char="")
            .with_columns(speaker="cvcy-" + pl.col("client_id").str.slice(-cid_length))
            .with_columns(
                utterance=pl.col("speaker")
                + "-"
                + pl.col("path").str.extract("([0-9]+)")
            )
            .select(["speaker", "utterance", "sentence", "path"])
            .collect()
        )

        # Loop through each file and convert from mp3 to wav
        converted_paths = []
        for row in tqdm(df.rows(named=True), desc="Converting files"):
            new_path = output_path / "clips" / f"{row['utterance']}.wav"
            converted_paths.append(str(new_path))
            convert_file(
                input_path / "clips" / row["path"],
                new_path,
            )

        df = df.with_columns(
            pl.Series("path", converted_paths), pl.lit("cy").alias("lang")
        )

        # Select only the stuff we need and write to a csv file
        df.select(["speaker", "utterance", "path", "lang", "sentence"]).write_csv(
            output_path / f"{split}.csv"
        )

    # Combine all splits and write to csv
    dfs = [pl.read_csv(output_path / f"{split}.csv") for split in DATASET_SPLITS]
    pl.concat(dfs).write_csv(output_path / "all.csv")


def determine_cid_length(input_path: Path) -> int:
    # Check if we can reduce the speakers to a five letter combination
    dfs = [
        pl.read_csv(input_path / f"{split}.tsv", separator="\t", quote_char="")
        for split in DATASET_SPLITS
    ]
    combined = pl.concat(dfs)

    unique_speakers = combined["client_id"].unique()
    for length in range(4, len(combined["client_id"][0]) + 1):
        shortened = {cid[-length:] for cid in unique_speakers}
        if len(shortened) == len(unique_speakers):
            return length

    raise ValueError("Unable to resolve a unique set of speakers")


def convert_file(input_path: Path, output_path: Path, overwrite: bool = False) -> bool:
    """Converts the CV .mp3 file provided to a mono channel, 16kHz wav file. If
    the file exists and overwrite is set to false nothing happens."""
    if not overwrite and output_path.exists():
        return

    # Set up transformer
    tf = sox.Transformer()
    tf.convert(samplerate=16_000, n_channels=1, bitdepth=16)

    # If parent folder doesn't exist, make
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Write converted file to disk
    status, _, _ = tf.build(
        input_filepath=input_path,
        output_filepath=output_path,
        return_output=True,
    )
    return status
