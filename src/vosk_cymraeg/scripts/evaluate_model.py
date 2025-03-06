import argparse
import logging
from io import TextIOWrapper
from pathlib import Path
from typing import Callable

import evaluate
import polars as pl
from rich.logging import RichHandler
from universal_edit_distance import (
    character_mean_error_rate,
    word_mean_error_rate,
)

from vosk_cymraeg.normalisation import normalise_sentence

_logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-results", required=True, type=argparse.FileType(), nargs="+"
    )
    parser.add_argument("--normalise", action="store_true")
    args = parser.parse_args()

    dataset_hashes = {Path(file.name).stem.split("_")[-1] for file in args.test_results}
    if len(dataset_hashes) == 1:
        _logger.info(
            f"Test set hash validation passed. All datasets contained hash '{list(dataset_hashes)[0]}'"
        )
    else:
        hash_str = ", ".join(
            [f"'{dataset_hash}'" for dataset_hash in sorted(dataset_hashes)]
        )
        _logger.error(
            f"Found multiple conflicting test set hashes ({hash_str}). This can result in inconsistent meaningless results. Please ensure that the models have been tested on the same data."
        )
        return

    _logger.info("Loading metrics 'wer' and 'cer'")
    metrics = {"wer": evaluate.load("wer"), "cer": evaluate.load("cer")}

    # Splits that we are using (this is slightly cursed)
    splits: dict[str, Callable[[pl.DataFrame], pl.LazyFrame]] = {
        "all": lambda results: results.lazy(),
        "cy": lambda results: results.lazy().filter(pl.col("lang") == "cy"),
        "en": lambda results: results.lazy().filter(pl.col("lang") == "en"),
        "read-speech": lambda results: results.lazy().filter(
            pl.col("speaker").str.starts_with("cvcy")
        ),
        "spon-speech": lambda results: results.lazy().filter(
            ~pl.col("speaker").str.starts_with("cvcy")
        ),
        "spon-speech-cy": lambda results: results.lazy().filter(
            ~pl.col("speaker").str.starts_with("cvcy") & (pl.col("lang") == "cy")
        ),
        "spon-speech-en": lambda results: results.lazy().filter(
            ~pl.col("speaker").str.starts_with("cvcy") & (pl.col("lang") == "en")
        ),
        "btb": lambda results: results.lazy().filter(
            pl.col("speaker").str.starts_with("btb")
        ),
        "cvcy": lambda results: results.lazy().filter(
            pl.col("speaker").str.starts_with("cvcy")
        ),
        "lla": lambda results: results.lazy().filter(
            pl.col("speaker").str.starts_with("lla")
        ),
        "lla-en": lambda results: results.lazy().filter(
            pl.col("speaker").str.starts_with("lla") & (pl.col("lang") == "en")
        ),
        "lla-cy": lambda results: results.lazy().filter(
            pl.col("speaker").str.starts_with("lla") & (pl.col("lang") == "cy")
        ),
    }

    summary = pl.concat(
        [
            get_summary_for_model(
                test_result, metrics, splits, normalise=args.normalise
            )
            for test_result in args.test_results
        ]
    ).sort(["set", "model"])
    _logger.info(summary)
    summary.write_clipboard()
    _logger.info("Table written to clipboard")


def get_summary_for_model(
    file: TextIOWrapper,
    metrics: dict[str, evaluate.EvaluationModule],
    splits: dict[str, Callable[[pl.DataFrame], pl.LazyFrame]],
    normalise: bool = False,
) -> pl.DataFrame:
    model_name = "_".join(Path(file.name).stem.split("_")[0:-1])
    _logger.info(f"Evaluating results for model {model_name}")

    # This needs to be moved to the eval script
    results = (
        pl.read_csv(file)
        .filter(pl.col("sentence").str.strip_chars() != "")
        .with_columns(pl.col("speaker").str.split("-").first().alias("dataset"))
    )

    if normalise:
        results = results.with_columns(
            pl.col("sentence").map_elements(normalise_sentence, pl.String),
            pl.col("transcription").map_elements(normalise_sentence, pl.String),
        )

    # print(results)
    summary = pl.DataFrame(
        [
            {"set": name, **run_evaluation(split(results), metrics)}
            for name, split in splits.items()
        ]
    ).insert_column(0, pl.lit(model_name).alias("model"))
    return summary


def run_evaluation(
    data: pl.LazyFrame, metrics: dict[str, evaluate.EvaluationModule]
) -> dict[str, float]:
    # Actually collect the data
    data = data.collect()
    results = {
        name: metric.compute(
            predictions=data["transcription"], references=data["sentence"]
        )
        for name, metric in metrics.items()
    }
    results["uwer"] = word_mean_error_rate(data["transcription"], data["sentence"])
    results["ucer"] = character_mean_error_rate(data["transcription"], data["sentence"])
    return results
