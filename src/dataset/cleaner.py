import logging
import pandas as pd

logger = logging.getLogger(__name__)


def clean_papers(all_papers: list[dict], min_abstract_length: int = 100) -> pd.DataFrame:
    """Clean a list of paper dictionaries by removing entries with missing or short abstracts.

    Args:
        all_papers: List of dictionaries, each representing a paper with at least
            'abstract', 'title', and 'category' keys.
        min_abstract_length: Minimum number of characters an abstract must have
            to be kept. Defaults to 100.

    Returns:
        A cleaned DataFrame with an additional 'abstract_length' column.
    """
    df = pd.DataFrame(all_papers)

    logger.info("Dataset before cleaning:")
    logger.info("Total papers: %d", len(df))
    logger.info("Papers with abstracts: %d", df["abstract"].notna().sum())

    missing_abstracts = df["abstract"].isna().sum()
    if missing_abstracts > 0:
        logger.warning("%d papers have missing abstracts", missing_abstracts)
        df = df.dropna(subset=["abstract"])

    df["abstract_length"] = df["abstract"].str.len()
    df = df[df["abstract_length"] >= min_abstract_length].copy()

    logger.info("Dataset after cleaning:")
    logger.info("Total papers: %d", len(df))
    logger.info("Average abstract length: %.0f characters", df["abstract_length"].mean())
    logger.info("Papers per category:\n%s", df["category"].value_counts().sort_index())

    return df
