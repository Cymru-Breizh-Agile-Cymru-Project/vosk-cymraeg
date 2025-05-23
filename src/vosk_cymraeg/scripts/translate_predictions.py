import argparse
import polars as pl
import datasets

from tqdm import tqdm

from transformers import pipeline


def main() -> None:
    test_df = get_dataset("prvInSpace/evals-kaldi-full-model")

    model_name = "DewiBrynJones/nllb-200-1.3B-ft-cym-to-eng"
    translator = pipeline("translation", model=model_name)

    def get_translation(text: str) -> str:
        return translator(text, src_lang="cym_Latn", tgt_lang="eng_Latn")[0][
            "translation_text"
        ]

    with tqdm(total=len(test_df)) as pbar:
        test_df = test_df.with_columns(
            pl.col("prediction")
            .map_elements(w_pbar(pbar, get_translation), pl.String)
            .alias("predicted_translation")
        )
    test_df.write_parquet("results/translation.parquet")


def get_dataset(path: str) -> pl.DataFrame:
    covost_df = get_covost_df()
    return (
        datasets.load_dataset(path, split="test")
        .to_polars()
        .rename({"transcription": "prediction"})
        .with_columns(pl.col("utterance").str.split("-").list.get(-1).alias("id"))
        .join(covost_df.select(["id", "translation"]), on="id", how="inner")
    )


def get_covost_df() -> pl.DataFrame:
    covost: datasets.DatasetDict = datasets.load_dataset(
        "facebook/covost2", "cy_en", data_dir="data/raw/cv/4/cy/"
    )
    return pl.concat([ds.remove_columns("audio").to_polars() for ds in covost.values()])


def parse_args() -> argparse.Namespace:
    pass


def w_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return foo
