import logging

from dataset.downloader import collect_arxiv_papers
from dataset.cleaner import clean_papers
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
    
if __name__ == "__main__":
    main()
