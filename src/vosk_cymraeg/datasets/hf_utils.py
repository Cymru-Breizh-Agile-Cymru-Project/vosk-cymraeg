import math
from io import BytesIO
from pathlib import Path

import datasets
import soundfile as sf
import sox
from tqdm import tqdm


def dump_dataset_audio_files(
    ds: datasets.Dataset, output_path: Path, batch_size: int = 1000
):
    number_of_batches = math.ceil(len(ds) / batch_size)

    # Batch dumps all of the bytes in audio
    for batch in tqdm(
        ds.to_polars(batched=True, batch_size=batch_size),
        total=number_of_batches,
        desc="Converting batches of audio",
    ):
        for row in tqdm(
            batch.iter_rows(named=True),
            leave=False,
            total=len(batch),
            desc="Converting clips",
        ):
            file_path = output_path / "clips" / f"{row['utterance']}.wav"
            dump_bytes_to_file(row["audio"]["bytes"], file_path)


def dump_bytes_to_file(
    bytes: bytes, output_path: Path, overwrite: bool = False
) -> bool:
    if not overwrite and output_path.exists():
        return

    # Read audio data
    data, sample_rate = sf.read(BytesIO(bytes))

    # Set up transformer
    tf = sox.Transformer()
    tf.convert(samplerate=16_000, n_channels=1, bitdepth=16)

    # If parent folder doesn't exist, make
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Write converted file to disk
    status, _, _ = tf.build(
        input_array=data,
        sample_rate_in=sample_rate,
        output_filepath=output_path,
        return_output=True,
    )
    return status
