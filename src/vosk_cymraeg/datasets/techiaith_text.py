import datasets
import polars as pl


def load_techiaith_cofnodycynulliad_en_cy() -> pl.DataFrame:
    """Dataset containing translations from the Welsh parliament's website"""
    df = datasets.load_dataset(
        "techiaith/cofnodycynulliad_en-cy", split="train"
    ).to_polars()
    return pl.concat(
        [
            pl.DataFrame({"sentence": df["target"], "lang": "cy"}),
            pl.DataFrame({"sentence": df["source"], "lang": "en"}),
        ]
    )


def load_techiaith_legislation_gov_uk_en_cy() -> pl.DataFrame:
    """Dataset containing translations from legislation.gov.uk"""
    df = datasets.load_dataset(
        "techiaith/legislation-gov-uk_en-cy", split="train"
    ).to_polars()
    return pl.concat(
        [
            pl.DataFrame({"sentence": df["target"], "lang": "cy"}),
            pl.DataFrame({"sentence": df["source"], "lang": "en"}),
        ]
    )


def load_techiaith_llyw_cymru_en_cy_ogl() -> pl.DataFrame:
    """Dataset containing translations from llyw.cymru"""
    df = datasets.load_dataset(
        "techiaith/llyw-cymru-en-cy-ogl", split="train"
    ).to_polars()
    return pl.concat(
        [
            pl.DataFrame({"sentence": df["text_cy"], "lang": "cy"}),
            pl.DataFrame({"sentence": df["text_en"], "lang": "en"}),
        ]
    )


def load_str20tbl_tts_prompts_cy_en() -> pl.DataFrame:
    return (
        datasets.load_dataset("str20tbl/tts-prompts-cy-en", split="train")
        .to_polars()
        .rename({"Lang": "lang", "Sentence": "sentence"})
        .select(["sentence", "lang"])
    )
