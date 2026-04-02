import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def viz_embedding(
	df: pd.DataFrame,
	embeddings: list[list[float]],
	config: dict
	) -> None:

	colors = [cat['color'] for cat in config['dataset']['categories']]
	category_names = [cat['name'] for cat in config['dataset']['categories']]
	category_codes = [cat['code'] for cat in config['dataset']['categories']]

	pca = PCA(n_components=2)
	embeddings_2d = pca.fit_transform(embeddings)

	for i, (cat_code, cat_name, color) in enumerate(zip(category_codes, category_names, colors)):
		# Get papers from this category
		mask = df['category'] == cat_code
		cat_embeddings = embeddings_2d[mask]

		plt.scatter(cat_embeddings[:, 0], cat_embeddings[:, 1],
					c=color, label=cat_name, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

	plt.xlabel('First Principal Component', fontsize=12)
	plt.ylabel('Second Principal Component', fontsize=12)
	plt.title('500 arXiv Papers Across Five Computer Science Categories \
			\n(Real-world embeddings show overlapping clusters)',
			fontsize=14, fontweight='bold', pad=20)
	plt.legend(loc='best', fontsize=10)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()