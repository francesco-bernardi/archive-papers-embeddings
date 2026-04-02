import logging
import time

import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def embedd_papers(df: pd.DataFrame, model: str, batch_size: int) -> pd.DataFrame:
	logger.info("Loading SentenceTransformer model: %s", model)
	model = SentenceTransformer(model)
	logger.info("Model loaded successfully")

	logger.info("Encoding abstracts...")
	start_time = time.time()
	embeddings = model.encode(
		df["abstract"].tolist(),
		batch_size=batch_size,
		show_progress_bar=True,
		convert_to_numpy=True,
	)
	end_time = time.time()
	logger.info("Encoded %d abstracts in %.2f seconds", len(embeddings), end_time - start_time)
	logger.info("Embeddings shape: %s", embeddings.shape)

	df["embedding"] = embeddings.tolist()

	return df