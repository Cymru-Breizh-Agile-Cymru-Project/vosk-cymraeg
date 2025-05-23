import datasets
import polars as pl

from vosk_cymraeg.normalisation import normalise_sentence
import universal_edit_distance as ued
from scipy.stats.mstats import f_oneway


ACCENTS = ("De Ddwyrain", "Gogledd Orllewin")#, "De Orllewin")

def main() -> None:
    df_names = [
        "DewiBrynJones/evals-ca25-whisper-large-v3-ft-btb-cv-ca-cy",
        "DewiBrynJones/evals-ca25-whisper-large-v3-ft-btb-ca-cy",
    ]

    # Load datasets
    df_dict = {name: load_hf_dataset(name) for name in df_names}
    df_dict.update(load_kaldi_dataset("prvInSpace/evals-kaldi-full-model"))
    df_dict.update(load_kaldi_dataset("prvInSpace/evals-kaldi-bilingual"))
    df_dict.update(load_kaldi_dataset("prvInSpace/evals-kaldi-all-with-corpus"))
    df_dict.update(load_kaldi_dataset("prvInSpace/evals-kaldi-ner-full"))
    df_dict.update(load_kaldi_dataset("prvInSpace/evals-kaldi-ner-text-only"))

    res = pl.DataFrame([get_normalised_values(df) for df in df_dict.values()])
    print(res)
    lookup = res.mean().transpose(
        include_header=True, header_name="accent", column_names=["correction"]
    )

    def get_bias(model: str):
        tmp = df_dict[model]
        tmp = tmp.join(lookup, on="accent", how="left").with_columns(
            (pl.col("wer") / pl.col("correction")).alias("corrected_wer")
        )
        wer_pv = analyse_variance(tmp, "wer")
        cwer_pv = analyse_variance(tmp, "corrected_wer")
        return {"model": model, "bias": wer_pv, "bias_corrected": cwer_pv}
    
    bias_table = pl.DataFrame([
        get_bias(model)
        for model in df_dict
    ])
    print(bias_table)


def analyse_variance(df: pl.DataFrame, field: str):
    accent_data = {
        accent: df.filter(pl.col("accent") == accent)[field]
        for accent in df["accent"].unique()
        if accent in ACCENTS
    }
    print(f"{df[field].mean():.2%} Overall")
    for accent, data in sorted(accent_data.items(), key=lambda x: x[1].mean()):
        print(f"{data.mean():.2%}", f"{len(data):4d}", accent)

    res = f_oneway(*[v.to_list() for v in accent_data.values()])
    print(res)
    return res.pvalue


def load_hf_dataset(dataset: str) -> pl.DataFrame:
    df = (
        datasets.load_dataset(dataset, split="test")
        .to_polars()
        .drop("audio")
        .rename({"prediction": "transcription"})
        .with_columns(
            pl.col("sentence").map_elements(normalise_sentence, pl.String),
            pl.col("transcription").map_elements(normalise_sentence, pl.String),
        )
    )
    return (
        df.with_columns(
            pl.Series(
                "wer",
                ued.word_error_rate(df["transcription"], df["sentence"]),
            )
        )
        .drop_nans("wer")
        .filter(pl.col("wer").is_finite())
    )


def load_kaldi_dataset(dataset: str) -> pl.DataFrame:
    lla_test = (
        datasets.load_dataset("cymen-arfor/lleisiau-arfor", split="test_clean")
        .remove_columns("audio")
        .to_polars()
    )
    lla_test = lla_test.with_columns(pl.int_range(0, len(lla_test)).alias("id"))
    test_set = (
        pl.read_csv("data/processed/dataset/test.csv")
        .filter(pl.col("speaker").str.starts_with("lla"))
        .with_columns(
            pl.col("speaker").str.strip_prefix("lla-").cast(pl.Int64).alias("id")
        )
        .with_columns(pl.col("id") - pl.col("id").min())
    )

    test_set = test_set.join(lla_test.select(["id", "accent"]), on="id")
    to_test = datasets.load_dataset(dataset, split="test").to_polars()

    results_table = test_set.join(
        to_test.select(["speaker", "transcription"]), on="speaker", how="left"
    ).with_columns(
        pl.col("sentence").map_elements(normalise_sentence, pl.String),
        pl.col("transcription").map_elements(normalise_sentence, pl.String),
    )

    # print(results_table)
    return {
        dataset: results_table.with_columns(
            pl.Series(
                "wer",
                ued.word_error_rate(
                    results_table["transcription"], results_table["sentence"]
                ),
            )
        )
        .drop_nans("wer")
        .filter(pl.col("wer").is_finite())
    }


def get_normalised_values(results_table: pl.DataFrame) -> dict[str, float]:
    overall_wer = results_table["wer"].mean()
    return {
        accent: results_table.filter(pl.col("accent") == accent)["wer"].mean()
        / overall_wer
        for accent in results_table["accent"].unique()
        if accent  in ACCENTS
    }


if __name__ == "__main__":
    main()
