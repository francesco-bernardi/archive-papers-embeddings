import pandas as pd


def create_vector_db_data(df: pd.DataFrame) -> list[dict]:
	vector_db_data = []
	for idx, row in df.iterrows():
		vector_db_data.append({
			'id': row['arxiv_id'],
			'embedding': row['embedding'], 
			'metadata': {
				'title': row['title'],
				'abstract': row['abstract'][:500], 
				'authors': ', '.join(row['authors'][:3]), 
				'category': row['category'],
				'published': str(row['published'])
			}
		})
	return vector_db_data