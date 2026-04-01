import logging
from dataset.downloader import collect_arxiv_papers
from dataset.cleaner import clean_papers

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_PAPERS = 100
CATEGORIES = [
    ('cs.LG', 'Machine Learning'),
    ('cs.CV', 'Computer Vision'),
    ('cs.CL', 'Computational Linguistics'),
    ('cs.DB', 'Databases'),
    ('cs.SE', 'Software Engineering'),
]


def collect_all_papers() -> list[dict]:
    all_papers = []
    for category_code, category_name in CATEGORIES:
        logger.info("Collecting papers from %s (%s)...", category_name, category_code)
        papers = collect_arxiv_papers(category_code, max_results=MAX_PAPERS)
        all_papers.extend(papers)
        logger.info("Collected %d papers", len(papers))
    return all_papers


def print_samples(all_papers: list[dict]) -> None:
    separator = "=" * 80
    logger.info("\n%s\nSAMPLE PAPERS\n%s", separator, separator)
    for i, (_, category_name) in enumerate(CATEGORIES):
        paper = all_papers[i * MAX_PAPERS]
        logger.info("%s | %s | %s...", category_name, paper['title'], paper['abstract'][:150])


def main() -> None:
    all_papers = collect_all_papers()
    logger.info("Total papers collected: %d", len(all_papers))
    print_samples(all_papers)
    df = clean_papers(all_papers)
    logger.info(df.head())


if __name__ == "__main__":
    main()
