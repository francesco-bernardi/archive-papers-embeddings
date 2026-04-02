import logging
import json

import numpy as np

from pathlib import Path
from dataset.downloader import collect_arxiv_papers
from dataset.cleaner import clean_papers
from dataset.vect_db import create_vector_db_data
from model.inference import embedd_papers
from utils.viz import viz_embedding
from utils.utils import load_config

config = load_config()

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

MAX_PAPERS = config["dataset"]["max_papers_per_category"]
CATEGORIES = config["dataset"]["categories"]
EXPERIMENTS = config["experiments"]
SAVE_DIR = Path(config["output"]["save_dir"])
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def collect_all_papers() -> list[dict]:
    all_papers = []
    for category in CATEGORIES:
        logger.info("Collecting papers from %s (%s)...", category['name'], category['code'])
        papers = collect_arxiv_papers(category['code'], max_results=MAX_PAPERS)
        all_papers.extend(papers)
        logger.info("Collected %d papers", len(papers))
    return all_papers

def print_samples(all_papers: list[dict]) -> None:
    separator = "=" * 80
    logger.info("\n%s\nSAMPLE PAPERS\n%s", separator, separator)
    for i, category in enumerate(CATEGORIES):
        paper = all_papers[i * MAX_PAPERS]
        logger.info("%s | %s | %s...", category['name'], paper['title'], paper['abstract'][:150])

def main() -> None:
    all_papers = collect_all_papers()
    logger.info("Total papers collected: %d", len(all_papers))

    df = clean_papers(all_papers)

    for exp_name, exp_config in EXPERIMENTS.items():
        logger.info("Running experiment: %s", exp_name)
        df_result = embedd_papers(
            df.copy(), 
            exp_config["model"], 
            exp_config["batch_size"]
        )

        viz_embedding(
            df_result, 
            df_result['embedding'].tolist(), 
            config=config
        )

        df_metadata = df_result[
            [
                'title', 
                'abstract', 
                'authors', 
                'published', 
                'category', 
                'arxiv_id', 
                'abstract_length'
            ]
        ]

        df_metadata.to_csv(SAVE_DIR / f'{exp_name}_metadata.csv', index=False)
        np.save(
            SAVE_DIR / f'{exp_name}_embeddings.npy', 
            df_result['embedding'].tolist()
        )

        vector_db_data = create_vector_db_data(df_result)
        with open(SAVE_DIR / f'{exp_name}_vector_db_data.json', 'w') as f:
            json.dump(vector_db_data, f, indent=2)

if __name__ == "__main__":
    main()
