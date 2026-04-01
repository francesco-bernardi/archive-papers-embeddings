import arxiv

# Create the arXiv client once and reuse it
# This is recommended by the arxiv package to respect rate limits
client = arxiv.Client()

def collect_arxiv_papers(category, max_results=100):
    """
    Collect papers from arXiv by category.

    Parameters:
    -----------
    category : str
        arXiv category code (e.g., 'cs.LG', 'cs.CV')
    max_results : int
        Maximum number of papers to retrieve

    Returns:
    --------
    list of dict
        List of paper dictionaries containing title, abstract, authors, etc.
    """
    # Construct search query for the category
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in client.results(search):
        paper = {
            'title': result.title,
            'abstract': result.summary,
            'authors': [author.name for author in result.authors],
            'published': result.published,
            'category': category,
            'arxiv_id': result.entry_id.split('/')[-1]
        }
        papers.append(paper)

    return papers