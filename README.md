# 1. Understanding, Generating, and Visualizing Embeddings

[***Taken From***](https://www.dataquest.io/blog/understanding-generating-and-visualizing-embeddings/)

Imagine you're searching through a massive library of data science papers looking for content about "cleaning messy datasets." A traditional keyword search returns papers that literally contain those exact words. But it completely misses an excellent paper about "handling missing values and duplicates" and another about "data validation techniques." Even though these papers teach exactly what you're looking for, you'll never see them because they're using different words.

This is the fundamental problem with keyword-based searches: they match words, not meaning. When you search for "neural network training," it won't connect you to papers about "optimizing deep learning models" or "improving model convergence," despite these being essentially the same topic.

**Embeddings** solve this by teaching machines to understand meaning instead of just matching text. And if you're serious about building AI systems, generating embeddings is a fundamental concept you need to master.

## What Are Embeddings?

Embeddings are numerical representations that capture semantic meaning. Instead of treating text as a collection of words to match, embeddings convert text into vectors (a list of numbers) where similar meanings produce similar vectors. Think of it like translating human language into a mathematical language that computers can understand and compare.

When we convert two pieces of text that mean similar things into embeddings, those embedding vectors will be mathematically close to each other in the **embedding space**. Think of the embedding space as a multi-dimensional map where meaning determines location. Papers about machine learning will cluster together. Papers about data cleaning will form their own group. And papers about data visualization? They'll gather in a completely different region. In a moment, we'll create a visualization that clearly demonstrates this.

## Setting Up Your Environment

Before we start working directly with embeddings, let's install the libraries we'll need. We'll use `sentence-transformers` from [Hugging Face](https://huggingface.co/sentence-transformers) to generate embeddings, `sklearn` for dimensionality reduction, `matplotlib` for visualization, and `numpy` to handle the numerical arrays we'll be working with.

This tutorial was developed using Python 3.12.12 with the following library versions. You can use these exact versions for guaranteed compatibility, or install the latest versions (which should work just fine):

```bash
# Developed with: Python 3.12.12
# sentence-transformers==5.1.1
# scikit-learn==1.6.1
# matplotlib==3.10.0
# numpy==2.0.2

pip install sentence-transformers scikit-learn matplotlib numpy
```

Run the command above in your terminal to install all required libraries. This will work whether you're using a Python script, Jupyter notebook, or any other development environment.

For this tutorial series, we'll work with research paper abstracts from [arXiv.org](http://arxiv.org/), a repository where researchers publish cutting-edge AI and machine learning papers. If you're building AI systems, [arXiv](https://arxiv.org/) is a great resource to have. It's where you'll find the latest research on new architectures, techniques, and approaches that can help you implement the latest techniques in your projects.

> arXiv is pronounced as "archive" because the X represents the Greek letter chi ⟨χ⟩

For this tutorial, we've manually created 12 abstracts for papers spanning machine learning, data engineering, and data visualization. These abstracts are stored directly in our code as Python strings, keeping things simple for now. We'll work with APIs and larger datasets in the next tutorial to automate this process.

```python
# Abstracts from three data science domains
papers = [
    # Machine Learning Papers
    {
        'title': 'Building Your First Neural Network with PyTorch',
        'abstract': '''Learn how to construct and train a neural network from scratch using PyTorch. This paper covers the fundamentals of defining layers, activation functions, and forward propagation. You'll build a multi-layer perceptron for classification tasks, understand backpropagation, and implement gradient descent optimization. By the end, you'll have a working model that achieves over 90% accuracy on the MNIST dataset.'''
    },
    {
        'title': 'Preventing Overfitting: Regularization Techniques Explained',
        'abstract': '''Overfitting is one of the most common challenges in machine learning. This guide explores practical regularization methods including L1 and L2 regularization, dropout layers, and early stopping. You'll learn how to detect overfitting by monitoring training and validation loss, implement regularization in both scikit-learn and TensorFlow, and tune regularization hyperparameters to improve model generalization on unseen data.'''
    },
    {
        'title': 'Hyperparameter Tuning with Grid Search and Random Search',
        'abstract': '''Selecting optimal hyperparameters can dramatically improve model performance. This paper demonstrates systematic approaches to hyperparameter optimization using grid search and random search. You'll learn how to define hyperparameter spaces, implement cross-validation during tuning, and use scikit-learn's GridSearchCV and RandomizedSearchCV. We'll compare both methods and discuss when to use each approach for efficient model optimization.'''
    },
    {
        'title': 'Transfer Learning: Using Pre-trained Models for Image Classification',
        'abstract': '''Transfer learning lets you leverage pre-trained models to solve new problems with limited data. This paper shows how to use pre-trained convolutional neural networks like ResNet and VGG for custom image classification tasks. You'll learn how to freeze layers, fine-tune network weights, and adapt pre-trained models to your specific domain. We'll build a classifier that achieves high accuracy with just a few hundred training images.'''
    },

    # Data Engineering/ETL Papers
    {
        'title': 'Handling Missing Data: Strategies and Best Practices',
        'abstract': '''Missing data can derail your analysis if not handled properly. This comprehensive guide covers detection methods for missing values, statistical techniques for understanding missingness patterns, and practical imputation strategies. You'll learn when to use mean imputation, forward fill, and more sophisticated approaches like KNN imputation. We'll work through real datasets with missing values and implement robust solutions using pandas.'''
    },
    {
        'title': 'Data Validation Techniques for ETL Pipelines',
        'abstract': '''Building reliable data pipelines requires thorough validation at every stage. This paper teaches you how to implement data quality checks, define validation rules, and catch errors before they propagate downstream. You'll learn schema validation, outlier detection, and referential integrity checks. We'll build a validation framework using Great Expectations and integrate it into an automated ETL workflow for production data systems.'''
    },
    {
        'title': 'Cleaning Messy CSV Files: A Practical Guide',
        'abstract': '''Real-world CSV files are rarely clean and analysis-ready. This hands-on paper walks through common data quality issues: inconsistent formatting, duplicate records, invalid entries, and encoding problems. You'll master pandas techniques for standardizing column names, removing duplicates, handling date parsing errors, and dealing with mixed data types. We'll transform a messy CSV with multiple quality issues into a clean dataset ready for analysis.'''
    },
    {
        'title': 'Building Scalable ETL Workflows with Apache Airflow',
        'abstract': '''Apache Airflow helps you build, schedule, and monitor complex data pipelines. This paper introduces Airflow's core concepts including DAGs, operators, and task dependencies. You'll learn how to define pipeline workflows, implement retry logic and error handling, and schedule jobs for automated execution. We'll build a complete ETL pipeline that extracts data from APIs, transforms it, and loads it into a data warehouse on a daily schedule.'''
    },

    # Data Visualization Papers
    {
        'title': 'Creating Interactive Dashboards with Plotly Dash',
        'abstract': '''Interactive dashboards make data exploration intuitive and engaging. This paper teaches you how to build web-based dashboards using Plotly Dash. You'll learn to create interactive charts with dropdowns, sliders, and date pickers, implement callbacks for dynamic updates, and design responsive layouts. We'll build a complete dashboard for exploring sales data with filters, multiple chart types, and real-time updates.'''
    },
    {
        'title': 'Matplotlib Best Practices: Making Publication-Quality Plots',
        'abstract': '''Creating clear, professional visualizations requires attention to design principles. This guide covers matplotlib best practices for publication-quality plots. You'll learn about color palette selection, font sizing and typography, axis formatting, and legend placement. We'll explore techniques for reducing chart clutter, choosing appropriate chart types, and creating consistent styling across multiple plots for research papers and presentations.'''
    },
    {
        'title': 'Data Storytelling: Designing Effective Visualizations',
        'abstract': '''Good visualizations tell a story and guide viewers to insights. This paper focuses on the principles of visual storytelling and effective chart design. You'll learn how to choose the right visualization for your data, apply pre-attentive attributes to highlight key information, and structure narratives through sequential visualizations. We'll analyze both effective and ineffective visualizations, discussing what makes certain design choices successful.'''
    },
    {
        'title': 'Building Real-Time Visualization Streams with Bokeh',
        'abstract': '''Visualizing streaming data requires specialized techniques and tools. This paper demonstrates how to create real-time updating visualizations using Bokeh. You'll learn to implement streaming data sources, update plots dynamically as new data arrives, and optimize performance for continuous updates. We'll build a live monitoring dashboard that displays streaming sensor data with smoothly updating line charts and real-time statistics.'''
    }
]

print(f"Loaded {len(papers)} paper abstracts")
print(f"Topics covered: Machine Learning, Data Engineering, and Data Visualization")
```
```
Loaded 12 paper abstracts
Topics covered: Machine Learning, Data Engineering, and Data Visualization
```

## Generating Your First Embeddings

Now let's transform these paper abstracts into embeddings. We'll use a pre-trained model from the [`sentence-transformers` library](https://huggingface.co/sentence-transformers) called `all-MiniLM-L6-v2`. We're using this model because it's fast and efficient for learning purposes, perfect for understanding the core concepts. In our next tutorial, we'll explore more recent production-grade models used in real-world applications.

The model will convert each abstract into an n-dimensional vector, where the value of n depends on the specific model architecture. Different embedding models produce vectors of different sizes. Some models create compact 128-dimensional embeddings, while others produce larger 768 or even 1024-dimensional vectors. Generally, larger embeddings can capture more nuanced semantic information, but they also require more computational resources and storage space.

Think of each dimension in the vector as capturing some aspect of the text's meaning. Maybe one dimension responds strongly to "machine learning" concepts, another to "data cleaning" terminology, and another to "visualization" language. The model learned these representations automatically during training.

Let's see what dimensionality our specific model produces.

```python
from sentence_transformers import SentenceTransformer

# Load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract just the abstracts for embedding
abstracts = [paper['abstract'] for paper in papers]

# Generate embeddings for all abstracts
embeddings = model.encode(abstracts)

# Let's examine what we've created
print(f"Shape of embeddings: {embeddings.shape}")
print(f"Each abstract is represented by a vector of {embeddings.shape[1]} numbers")
print(f"\nFirst few values of the first embedding:")
print(embeddings[0][:10])
```
```
Shape of embeddings: (12, 384)
Each abstract is represented by a vector of 384 numbers

First few values of the first embedding:
[-0.06071806 -0.13064863  0.00328695 -0.04209436 -0.03220841  0.02034248
  0.0042156  -0.01300791 -0.1026612  -0.04565621]
```

Perfect! We now have 12 embeddings, one for each paper abstract. Each embedding is a 384-dimensional vector, represented as a NumPy array of floating-point numbers.

These numbers might look random at first, but they encode meaningful information about the semantic content of each abstract. When we want to find similar documents, we measure the **cosine similarity** between their embedding vectors. Cosine similarity looks at the angle between vectors. Vectors pointing in similar directions (representing similar meanings) have high cosine similarity, even if their magnitudes differ. In a later tutorial, we'll compute vector similarity using cosine, Euclidean, and dot-product methods to compare different approaches.

Before we move on, let's verify we can retrieve the original text:

```python
# Let's look at one paper and its embedding
print("Paper title:", papers[0]['title'])
print("\nAbstract:", papers[0]['abstract'][:100] + "...")
print("\nEmbedding shape:", embeddings[0].shape)
print("Embedding type:", type(embeddings[0]))
```
```
Paper title: Building Your First Neural Network with PyTorch

Abstract: Learn how to construct and train a neural network from scratch using PyTorch. This paper covers the ...

Embedding shape: (384,)
Embedding type: <class 'numpy.ndarray'>
```

Great! We can still access the original paper text alongside its embedding. Throughout this tutorial, we'll work with these embeddings while keeping track of which paper each one represents.

## Making Sense of High-Dimensional Spaces

We now have 12 vectors, each with 384 dimensions. But here's the issue: humans can't visualize 384-dimensional space. We struggle to imagine even four dimensions! To understand what our embeddings have learned, we need to reduce them to two dimensions so that we can plot them on a graph.

This is where **dimensionality reduction** is a good skill to have. We'll use **Principal Component Analysis (PCA)**, a technique we can use to find the two most important dimensions (the ones that capture the most variation in our data). It's like taking a 3D object and finding the best angle to photograph it in 2D while preserving as much information as possible.

While we're definitely going to lose some detail during this compression, our original 384-dimensional embeddings capture rich, nuanced information about semantic meaning. When we squeeze them down to 2D, some subtleties are bound to get lost. But the major patterns (which papers belong to which topic) will still be clearly visible.

```python
from sklearn.decomposition import PCA

# Reduce embeddings from 384 dimensions to 2 dimensions
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

print(f"Original embedding dimensions: {embeddings.shape[1]}")
print(f"Reduced embedding dimensions: {embeddings_2d.shape[1]}")
print(f"\nVariance explained by these 2 dimensions: {pca.explained_variance_ratio_.sum():.2%}")
```
```
Original embedding dimensions: 384
Reduced embedding dimensions: 2

Variance explained by these 2 dimensions: 41.20%
```

The variance explained tells us how much of the variation in the original data is preserved in these 2 dimensions. Think of it this way: if all our papers were identical, they'd have zero variance. The more different they are, the more variance. We've kept about 41% of that variation, which is plenty to see the major patterns. The clustering itself depends on whether papers use similar vocabulary, not on how much variance we've retained. So even though 41% might seem relatively low, the major patterns separating different topics will still be very clear in our embedding visualization.

## Understanding Our Tutorial Topics

Before we create our embeddings visualization, let's see how the 12 papers are organized by topic. This will help us understand the patterns we're about to see in the embeddings:

```python
# Print papers grouped by topic
print("=" * 80)
print("PAPER REFERENCE GUIDE")
print("=" * 80)

topics = [
    ("Machine Learning", list(range(0, 4))),
    ("Data Engineering/ETL", list(range(4, 8))),
    ("Data Visualization", list(range(8, 12)))
]

for topic_name, indices in topics:
    print(f"\n{topic_name}:")
    print("-" * 80)
    for idx in indices:
        print(f"  Paper {idx+1}: {papers[idx]['title']}")
```
```
================================================================================
PAPER REFERENCE GUIDE
================================================================================

Machine Learning:
--------------------------------------------------------------------------------
  Paper 1: Building Your First Neural Network with PyTorch
  Paper 2: Preventing Overfitting: Regularization Techniques Explained
  Paper 3: Hyperparameter Tuning with Grid Search and Random Search
  Paper 4: Transfer Learning: Using Pre-trained Models for Image Classification

Data Engineering/ETL:
--------------------------------------------------------------------------------
  Paper 5: Handling Missing Data: Strategies and Best Practices
  Paper 6: Data Validation Techniques for ETL Pipelines
  Paper 7: Cleaning Messy CSV Files: A Practical Guide
  Paper 8: Building Scalable ETL Workflows with Apache Airflow

Data Visualization:
--------------------------------------------------------------------------------
  Paper 9: Creating Interactive Dashboards with Plotly Dash
  Paper 10: Matplotlib Best Practices: Making Publication-Quality Plots
  Paper 11: Data Storytelling: Designing Effective Visualizations
  Paper 12: Building Real-Time Visualization Streams with Bokeh
```

Now that we know which tutorials belong to which topic, let's visualize their embeddings.

## Visualizing Embeddings to Reveal Relationships

We're going to create a scatter plot where each point represents one paper abstract. We'll color-code them by topic so we can see how the embeddings naturally group similar content together.

```python
import matplotlib.pyplot as plt
import numpy as np

# Create the visualization
plt.figure(figsize=(8, 6))

# Define colors for different topics
colors = ['#0066CC', '#CC0099', '#FF6600']
categories = ['Machine Learning', 'Data Engineering/ETL', 'Data Visualization']

# Create color mapping for each paper
color_map = []
for i in range(12):
    if i < 4:
        color_map.append(colors[0])  # Machine Learning
    elif i < 8:
        color_map.append(colors[1])  # Data Engineering
    else:
        color_map.append(colors[2])  # Data Visualization

# Plot each paper
for i, (x, y) in enumerate(embeddings_2d):
    plt.scatter(x, y, c=color_map[i], s=275, alpha=0.7, edgecolors='black', linewidth=1)
    # Add paper numbers as labels
    plt.annotate(str(i+1), (x, y), fontsize=10, fontweight='bold',
                ha='center', va='center')

plt.xlabel('First Principal Component', fontsize=14)
plt.ylabel('Second Principal Component', fontsize=14)
plt.title('Paper Embeddings from Three Data Science Topics\n(Papers close together have similar semantic meaning)',
          fontsize=15, fontweight='bold', pad=20)

# Add a legend showing which colors represent which topics
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[i], markersize=12,
                              label=categories[i]) for i in range(len(categories))]
plt.legend(handles=legend_elements, loc='best', fontsize=12)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## What the Visualization Reveals About Semantic Similarity

Take a look at the visualization below that was generated using the code above. As you can see, the results are pretty striking! The embeddings have naturally organized themselves into three distinct regions based purely on semantic content.

Keep in mind that we deliberately chose papers from very distinct topics to make the clustering crystal clear. This is perfect for learning, but real-world datasets are messier. When you're working with papers that bridge multiple topics or have overlapping vocabulary, you'll see more gradual transitions between clusters rather than these sharp separations. We'll encounter that reality in the next tutorial when we work with hundreds of real arXiv papers.

![Paper Embeddings from Data Science Topics](https://www.dataquest.io/wp-content/uploads/2025/10/Paper-Embeddings-from-Data-Science-Topics.png.webp)

- **The Machine Learning cluster (blue, papers 1-4)** dominates the lower-left side of the plot. These four points sit close together because they all discuss neural networks, training, and model optimization. Look at papers 1 and 4. They're positioned very near each other despite one focusing on building networks from scratch and the other on transfer learning. The embedding model recognizes that they both use the core language of deep learning: layers, weights, training, and model architectures.
- **The Data Engineering/ETL cluster (magenta, papers 5-8)** occupies the upper portion of the plot. These papers share vocabulary around data quality, pipelines, and validation. Notice how papers 5, 6, and 7 form a tight grouping. They all discuss data quality issues using terms like "missing values," "validation," and "cleaning." Paper 8 (about Airflow) sits slightly apart from the others, which makes sense: while it's definitely about data engineering, it focuses more on workflow orchestration than data quality, giving it a slightly different semantic fingerprint.
- **The Data Visualization cluster (orange, papers 9-12)** is gathered on the lower-right side. These four papers are packed close together because they all use visualization-specific vocabulary: "charts," "dashboards," "colors," and "interactive elements." The tight clustering here shows just how distinct visualization terminology is from both ML and data engineering language.

What's remarkable is the clear separation between all three clusters. The distance between the ML papers on the left and the visualization papers on the right tells us that these topics use fundamentally different vocabularies. There's minimal semantic overlap between "neural networks" and "dashboards," so they end up far apart in the **embedding space**.

### How the Model Learned to Understand Meaning

The [`all-MiniLM-L6-v2` embedding model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) was trained on millions of text pairs, learning which words tend to appear together. When it sees a tutorial full of words like "layers," "training," and "optimization," it produces an embedding vector that's mathematically similar to other texts with that same vocabulary pattern. The clustering emerges naturally from those learned associations.

## Why This Matters for Your Work as an AI Engineer

Embeddings are foundational to the modern AI systems you'll build as an AI Engineer. Let's look at how embeddings enable the core technologies you'll work with:

1. **Building Intelligent Search Systems**
	Traditional keyword search has a fundamental limitation: it can only find exact matches. If a user searches for "handling null values," they won't find documents about "missing data strategies" or "imputation techniques," even though these are exactly what they need. Embeddings solve this by understanding semantic similarity. When you embed both the search query and your documents, you can find relevant content based on meaning rather than word matching. The result is a search system that actually understands what you're looking for.
2. **Working with Vector Databases**
	Vector databases are specialized databases that are built to store and query embeddings efficiently. Instead of SQL queries that match exact values, vector databases let you ask "find me all documents similar to this one" and get results ranked by semantic similarity. They're optimized for the mathematical operations that embeddings require, like calculating distances between high-dimensional vectors, which makes them essential infrastructure for AI applications. Modern systems often use hybrid search approaches that combine semantic similarity with traditional keyword matching to get the best of both worlds.
3. **Implementing Retrieval-Augmented Generation (RAG)**
	RAG systems are one of the most powerful patterns in modern AI engineering. Here's how they work: you embed a large collection of documents (like company documentation, research papers, or knowledge bases). When a user asks a question, you embed their question and use that embedding to find the most relevant documents from your collection. Then you pass those documents to a language model, which generates an informed answer grounded in your specific data. Embeddings make the retrieval step possible because they're how the system knows which documents are relevant to the question.
4. **Creating AI Agents with Long-Term Memory**
	AI agents that can remember past interactions and learn from experience need a way to store and retrieve relevant memories. Embeddings enable this. When an agent has a conversation or completes a task, you can embed the key information and store it in a vector database. Later, when the agent encounters a similar situation, it can retrieve relevant past experiences by finding embeddings close to the current context. This gives agents the ability to learn from history and make better decisions over time. In practice, long-term agent memory often uses similarity thresholds and time-weighted retrieval to prevent irrelevant or outdated information from being recalled.

These four applications (search, vector databases, RAG, and AI agents) are foundational tools for any aspiring AI Engineer's toolkit. Each builds on embeddings as a core technology. Understanding how embeddings capture semantic meaning is the first step toward building production-ready AI systems.

## Advanced Topics to Explore

As you continue learning about embeddings, you'll encounter several advanced techniques that are widely used in production systems:

- **Multimodal Embeddings** allow you to embed different types of content (text, images, audio) into the same embedding space. This enables powerful cross-modal search capabilities, like finding images based on text descriptions or vice versa. Models like [CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) demonstrate how effective this approach can be.
- **Instruction-Tuned Embeddings** are models fine-tuned to better understand specific types of queries or instructions. These specialized models often outperform general-purpose embeddings for domain-specific tasks like legal document search or medical literature retrieval.
- **Quantization** reduces the precision of embedding values (from 32-bit floats to 8-bit integers, for example), which can dramatically reduce storage requirements and speed up similarity calculations with minimal impact on search quality. This becomes crucial when working with millions of embeddings.
- **Dimension Truncation** takes advantage of the fact that the most important information in embeddings is often concentrated in the first dimensions. By keeping only the first 256 dimensions of a 768-dimensional embedding, you can achieve significant efficiency gains while preserving most of the semantic information.

These techniques become increasingly important as you scale from prototype to production systems handling real-world data volumes.

## Building Toward Production Systems

You've now learned the following core foundational embedding concepts:

- Embeddings convert text into numerical vectors that capture meaning
- Similar content produces similar vectors
- These relationships can be visualized to understand how the model organizes information

But we've only worked with 12 handwritten paper abstracts. This is perfect for getting the core concept, but real applications need to handle hundreds or thousands of documents.

In the next tutorial, we'll scale up dramatically. You'll learn how to collect documents programmatically using APIs, generate embeddings at scale, and make strategic decisions about different embedding approaches.

You'll also face the practical challenges that come with real data: rate limits on APIs, processing time for large datasets, the tradeoff between embedding quality and speed, and how to handle edge cases like empty documents or very long texts. These considerations separate a learning exercise from a production system.

By the end of the next tutorial, you'll be equipped to build an embedding system that handles real-world data at scale. That foundation will prepare you for our final embeddings tutorial, where we'll implement similarity search and build a complete semantic search engine.

### Next Steps

For now, experiment with the code above:

- Try replacing one of the paper abstracts with content from your own learning.
	- Where does it appear on the visualization?
	- Does it cluster with one of our three topics, or does it land somewhere in between?
- Add a paper abstract that bridges two topics, like "Using Machine Learning to Optimize ETL Pipelines."
	- Does it position itself between the ML and data engineering clusters?
	- What does this tell you about how embeddings handle multi-topic content?
- Try changing the embedding model to see how it affects the visualization.
	- Models like `all-mpnet-base-v2` produce different dimensional embeddings.
	- Do the clusters become tighter or more spread out?
- Experiment with adding a completely unrelated abstract, like a cooking recipe or news article.
	- Where does it land relative to our three data science clusters?
	- How far away is it from the nearest cluster?

This hands-on exploration and experimentation will deepen your intuition about how embeddings work.

Ready to scale things up? In the next tutorial, we'll work with real arXiv data and build an embedding system that can handle thousands of papers. See you there!

---

**Key Takeaways**:

- Embeddings convert text into numerical vectors that capture semantic meaning
- Similar meanings produce similar vectors, enabling mathematical comparison of concepts
- Papers from different topics cluster separately because they use distinct vocabulary
- Dimensionality reduction (like PCA) helps visualize high-dimensional embeddings in 2D
- Embeddings power modern AI systems, including semantic search, vector databases, RAG, and AI agents


# 2. Generating Embeddings with APIs and Open Models

[***Taken From***](https://www.dataquest.io/blog/generating-embeddings-with-apis-and-open-models/)

In the previous tutorial, you learned that embeddings convert text into numerical vectors that capture semantic meaning. You saw how papers about machine learning, data engineering, and data visualization naturally clustered into distinct groups when we visualized their embeddings. That was the foundation.

But we only worked with 12 handwritten paper abstracts that we typed directly into our code. That approach works great for understanding core concepts, but it doesn't prepare you for real projects. Real applications require processing hundreds or thousands of documents, and you need to make strategic decisions about how to generate those embeddings efficiently.

This tutorial teaches you how to collect documents programmatically and generate embeddings using different approaches. You'll use the [arXiv API](https://info.arxiv.org/help/api/index.html) to gather 500 research papers, then generate embeddings using both local models and cloud services. By comparing these approaches hands-on, you'll understand the tradeoffs and be able to make informed decisions for your own projects.

These techniques form the foundation for production systems, but we're focusing on core concepts with a learning-sized dataset. A real system handling millions of documents would require batching strategies, streaming pipelines, and specialized vector databases. We'll touch on those considerations, but our goal here is to build your intuition about the embedding generation process itself.

## Setting Up Your Environment

Before we start collecting data, let's install the libraries we'll need. We'll use the `arxiv` library to access research papers programmatically, `pandas` for data manipulation, and the same embedding libraries from the previous tutorial.

This tutorial was developed using Python 3.12.12 with the following library versions. You can use these exact versions for guaranteed compatibility, or install the latest versions (which should work just fine):

```bash
# Developed with: Python 3.12.12
# sentence-transformers==5.1.2
# scikit-learn==1.6.1
# matplotlib==3.10.0
# numpy==2.0.2
# arxiv==2.2.0
# pandas==2.2.2
# cohere==5.20.0
# python-dotenv==1.1.1

pip install sentence-transformers scikit-learn matplotlib numpy arxiv pandas cohere python-dotenv
```

This tutorial works in any Python environment: Jupyter notebooks, Python scripts, VS Code, or your preferred IDE. Run the `pip` command above in your terminal before starting, then use the Python code blocks throughout this tutorial.

## Collecting Research Papers with the arXiv API

[arXiv](https://arxiv.org/) is a repository of over 2 million scholarly papers in physics, mathematics, computer science, and more. Researchers publish cutting-edge work here before it appears in journals, making it a valuable resource for staying current with AI and machine learning research. Best of all, arXiv provides a free API for programmatic access. While they do monitor usage and have some rate limits to prevent abuse, these limits are generous for learning and research purposes. Check their [Terms of Use](https://info.arxiv.org/help/api/tou.html) for current guidelines.

We'll use the arXiv API to collect 500 papers from five different computer science categories. This diversity will give us clear semantic clusters when we visualize or search our embeddings. The categories we'll use are:

- **cs.LG** (Machine Learning): Core ML algorithms, training methods, and theoretical foundations
- **cs.CV** (Computer Vision): Image processing, object detection, and visual recognition
- **cs.CL** (Computational Linguistics/NLP): Natural language processing and understanding
- **cs.DB** (Databases): Data storage, query optimization, and database systems
- **cs.SE** (Software Engineering): Development practices, testing, and software architecture

These categories use distinct vocabularies and will create well-separated clusters in our embedding space. Let's write a function to collect papers from specific arXiv categories:

```python
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

# Define the categories we want to collect from
categories = [
    ('cs.LG', 'Machine Learning'),
    ('cs.CV', 'Computer Vision'),
    ('cs.CL', 'Computational Linguistics'),
    ('cs.DB', 'Databases'),
    ('cs.SE', 'Software Engineering')
]

# Collect 100 papers from each category
all_papers = []
for category_code, category_name in categories:
    print(f"Collecting papers from {category_name} ({category_code})...")
    papers = collect_arxiv_papers(category_code, max_results=100)
    all_papers.extend(papers)
    print(f"  Collected {len(papers)} papers")

print(f"\nTotal papers collected: {len(all_papers)}")

# Let's examine the first paper from each category
separator = "=" * 80
print(f"\n{separator}", "SAMPLE PAPERS (one from each category)", f"{separator}", sep="\n")
for i, (_, category_name) in enumerate(categories):
    paper = all_papers[i * 100]
    print(f"\n{category_name}:")
    print(f"  Title: {paper['title']}")
    print(f"  Abstract (first 150 chars): {paper['abstract'][:150]}...")
```
```
Collecting papers from Machine Learning (cs.LG)...
  Collected 100 papers
Collecting papers from Computer Vision (cs.CV)...
  Collected 100 papers
Collecting papers from Computational Linguistics (cs.CL)...
  Collected 100 papers
Collecting papers from Databases (cs.DB)...
  Collected 100 papers
Collecting papers from Software Engineering (cs.SE)...
  Collected 100 papers

Total papers collected: 500

================================================================================
SAMPLE PAPERS (one from each category)
================================================================================

Machine Learning:
  Title: Dark Energy Survey Year 3 results: Simulation-based $w$CDM inference from weak lensing and galaxy clustering maps with deep learning. I. Analysis design
  Abstract (first 150 chars): Data-driven approaches using deep learning are emerging as powerful techniques to extract non-Gaussian information from cosmological large-scale struc...

Computer Vision:
  Title: Carousel: A High-Resolution Dataset for Multi-Target Automatic Image Cropping
  Abstract (first 150 chars): Automatic image cropping is a method for maximizing the human-perceived quality of cropped regions in photographs. Although several works have propose...

Computational Linguistics:
  Title: VeriCoT: Neuro-symbolic Chain-of-Thought Validation via Logical Consistency Checks
  Abstract (first 150 chars): LLMs can perform multi-step reasoning through Chain-of-Thought (CoT), but they cannot reliably verify their own logic. Even when they reach correct an...

Databases:
  Title: Are We Asking the Right Questions? On Ambiguity in Natural Language Queries for Tabular Data Analysis
  Abstract (first 150 chars): Natural language interfaces to tabular data must handle ambiguities inherent to queries. Instead of treating ambiguity as a deficiency, we reframe it ...

Software Engineering:
  Title: evomap: A Toolbox for Dynamic Mapping in Python
  Abstract (first 150 chars): This paper presents evomap, a Python package for dynamic mapping. Mapping methods are widely used across disciplines to visualize relationships among ...
```

The code above demonstrates how easy it is to collect papers programmatically. In just a few lines, we've gathered 500 recent research papers from five distinct computer science domains.

Take a look at your output when you run this code. You might notice something interesting: sometimes the same paper title appears under multiple categories. This happens because researchers often cross-list their papers in multiple relevant categories on arXiv. A paper about deep learning for natural language processing could legitimately appear in both Machine Learning (cs.LG) and Computational Linguistics (cs.CL). A paper about neural networks for image generation might be listed in both Machine Learning (cs.LG) and Computer Vision (cs.CV).

While our five categories are conceptually separate, there's naturally some overlap, especially between closely related fields. This real-world messiness is exactly what makes working with actual data more interesting than handcrafted examples. Your specific results will look different from ours because arXiv returns the most recently submitted papers, which change as new research is published.

## Preparing Your Dataset

Before generating embeddings, we need to clean and structure our data. Real-world datasets always have imperfections. Some papers might have missing abstracts, others might have abstracts that are too short to be meaningful, and we need to organize everything into a format that's easy to work with.

Let's use pandas to create a DataFrame and handle these data quality issues:

```python
import pandas as pd

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(all_papers)

print("Dataset before cleaning:")
print(f"Total papers: {len(df)}")
print(f"Papers with abstracts: {df['abstract'].notna().sum()}")

# Check for missing abstracts
missing_abstracts = df['abstract'].isna().sum()
if missing_abstracts > 0:
    print(f"\nWarning: {missing_abstracts} papers have missing abstracts")
    df = df.dropna(subset=['abstract'])

# Filter out papers with very short abstracts (less than 100 characters)
# These are often just placeholders or incomplete entries
df['abstract_length'] = df['abstract'].str.len()
df = df[df['abstract_length'] >= 100].copy()

print(f"\nDataset after cleaning:")
print(f"Total papers: {len(df)}")
print(f"Average abstract length: {df['abstract_length'].mean():.0f} characters")

# Show the distribution across categories
print("\nPapers per category:")
print(df['category'].value_counts().sort_index())

# Display the first few entries
separator = "=" * 80
print(f"\n{separator}", "FIRST 3 PAPERS IN CLEANED DATASET", f"{separator}", sep="\n")
for idx, row in df.head(3).iterrows():
    print(f"\n{idx+1}. {row['title']}")
    print(f"   Category: {row['category']}")
    print(f"   Abstract length: {row['abstract_length']} characters")
```
```
Dataset before cleaning:
Total papers: 500
Papers with abstracts: 500

Dataset after cleaning:
Total papers: 500
Average abstract length: 1337 characters

Papers per category:
category
cs.CL    100
cs.CV    100
cs.DB    100
cs.LG    100
cs.SE    100
Name: count, dtype: int64

================================================================================
FIRST 3 PAPERS IN CLEANED DATASET
================================================================================

1. Dark Energy Survey Year 3 results: Simulation-based $w$CDM inference from weak lensing and galaxy clustering maps with deep learning. I. Analysis design
   Category: cs.LG
   Abstract length: 1783 characters

2. Multi-Method Analysis of Mathematics Placement Assessments: Classical, Machine Learning, and Clustering Approaches
   Category: cs.LG
   Abstract length: 1519 characters

3. Forgetting is Everywhere
   Category: cs.LG
   Abstract length: 1150 characters
```

Data preparation matters because poor quality input leads to poor quality embeddings. By filtering out papers with missing or very short abstracts, we ensure that our embeddings will capture meaningful semantic content. In production systems, you'd likely implement more sophisticated quality checks, but this basic approach handles the most common issues.

## Strategy One: Local Open-Source Models

Now we're ready to generate embeddings. Let's start with local models using `sentence-transformers`, the same approach we used in the previous tutorial. The key advantage of local models is that everything runs on your own machine. There are no API costs, no data leaves your computer, and you have complete control over the embedding process.

We'll use `all-MiniLM-L6-v2` again for consistency, and we'll also demonstrate a larger model called `all-mpnet-base-v2` to show how different models produce different results:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# Load the same model from the previous tutorial
print("Loading all-MiniLM-L6-v2 model...")
model_small = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all abstracts
abstracts = df['abstract'].tolist()

print(f"Generating embeddings for {len(abstracts)} papers...")
start_time = time.time()

# The encode() method handles batching automatically
embeddings_small = model_small.encode(
    abstracts,
    show_progress_bar=True,
    batch_size=32  # Process 32 abstracts at a time
)

elapsed_time = time.time() - start_time

print(f"\nCompleted in {elapsed_time:.2f} seconds")
print(f"Embedding shape: {embeddings_small.shape}")
print(f"Each abstract is now a {embeddings_small.shape[1]}-dimensional vector")
print(f"Average time per abstract: {elapsed_time/len(abstracts):.3f} seconds")

# Add embeddings to our DataFrame
df['embedding_minilm'] = list(embeddings_small)
```
```
Loading all-MiniLM-L6-v2 model...
Generating embeddings for 500 papers...
Batches: 100%|██████████| 16/16 [01:05<00:00,  4.09s/it]

Completed in 65.45 seconds
Embedding shape: (500, 384)
Each abstract is now a 384-dimensional vector
Average time per abstract: 0.131 seconds
```

That was fast! On a typical laptop, we generated embeddings for 500 abstracts in about 65 seconds. Now let's try a larger, more powerful model to see the difference.

**Spoiler alert**: this will take several more minutes than the last one, so you may want to freshen up your coffee while it's running:

```python
# Load a larger (more dimensions) model
print("\nLoading all-mpnet-base-v2 model...")
model_large = SentenceTransformer('all-mpnet-base-v2')

print("Generating embeddings with larger model...")
start_time = time.time()

embeddings_large = model_large.encode(
    abstracts,
    show_progress_bar=True,
    batch_size=32
)

elapsed_time = time.time() - start_time

print(f"\nCompleted in {elapsed_time:.2f} seconds")
print(f"Embedding shape: {embeddings_large.shape}")
print(f"Each abstract is now a {embeddings_large.shape[1]}-dimensional vector")
print(f"Average time per abstract: {elapsed_time/len(abstracts):.3f} seconds")

# Add these embeddings to our DataFrame too
df['embedding_mpnet'] = list(embeddings_large)
```
```
Loading all-mpnet-base-v2 model...
Generating embeddings with larger model...
Batches: 100%|██████████| 16/16 [11:20<00:00, 30.16s/it]

Completed in 680.47 seconds
Embedding shape: (500, 768)
Each abstract is now a 768-dimensional vector
Average time per abstract: 1.361 seconds
```

Notice the differences between these two models:

- **Dimensionality:** The smaller model produces 384-dimensional embeddings, while the larger model produces 768-dimensional embeddings. More dimensions can capture more nuanced semantic information.
- **Speed:** The smaller model is about 10 times faster. For 500 papers, that's a difference of about 10 minutes. For thousands of documents, this difference becomes significant.
- **Quality:** Larger models generally produce higher-quality embeddings that better capture subtle semantic relationships. However, the smaller model is often good enough for many applications.

The key insight here is that local models give you flexibility. You can choose models that balance quality, speed, and computational resources based on your specific needs. For rapid prototyping, use smaller models. For production systems where quality matters most, use larger models.

## Visualizing Real-World Embeddings

In our previous tutorial, we saw beautifully separated clusters using handcrafted paper abstracts. Let's see what happens when we visualize embeddings from real arXiv papers. We'll use the same PCA approach to reduce our 384-dimensional embeddings down to 2D:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce embeddings from 384 dimensions to 2 dimensions
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_small)

print(f"Original embedding dimensions: {embeddings_small.shape[1]}")
print(f"Reduced embedding dimensions: {embeddings_2d.shape[1]}")
```
```
Original embedding dimensions: 384
Reduced embedding dimensions: 2
```

Now let's create a visualization showing how our 500 papers cluster by category:

```python
# Create the visualization
plt.figure(figsize=(12, 8))

# Define colors for different categories
colors = ['#C8102E', '#003DA5', '#00843D', '#FF8200', '#6A1B9A']
category_names = ['Machine Learning', 'Computer Vision', 'Comp. Linguistics', 'Databases', 'Software Eng.']
category_codes = ['cs.LG', 'cs.CV', 'cs.CL', 'cs.DB', 'cs.SE']

# Plot each category
for i, (cat_code, cat_name, color) in enumerate(zip(category_codes, category_names, colors)):
    # Get papers from this category
    mask = df['category'] == cat_code
    cat_embeddings = embeddings_2d[mask]

    plt.scatter(cat_embeddings[:, 0], cat_embeddings[:, 1],
                c=color, label=cat_name, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('500 arXiv Papers Across Five Computer Science Categories\n(Real-world embeddings show overlapping clusters)',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

![Visualization of embeddings for 500 arXiv Papers Across Five Computer Science Categories ](https://www.dataquest.io/wp-content/uploads/2025/11/500-arXiv-papers-clustered-by-category-1.png.webp)

The visualization above reveals an important aspect of real-world data. Unlike our handcrafted examples in the previous tutorial, where clusters were perfectly separated, these real arXiv papers show more overlap. You can see clear groupings, as well as papers that bridge multiple topics. For example, a paper about "deep learning for database query optimization" uses vocabulary from both machine learning and databases, so it might appear between those clusters.

This is exactly what you'll encounter in production systems. Real data is messy, topics overlap, and semantic boundaries are often fuzzy rather than sharp. The embeddings are still capturing meaningful relationships, but the visualization shows the complexity of actual research papers rather than the idealized examples we used for learning.

## Strategy Two: API-Based Embedding Services

Local models work great, but they require computational resources and you're responsible for managing them. API-based embedding services offer an alternative approach. You send your text to a cloud provider, they generate embeddings using their infrastructure, and they send the embeddings back to you.

We'll use Cohere's API for our main example because they offer a generous free trial tier that doesn't require payment information. This makes it perfect for learning and experimentation.

### Setting Up Cohere Securely

First, you'll need to create a free Cohere account and get an API key:

1. Visit [Cohere's registration page](https://dashboard.cohere.com/welcome/register)
2. Sign up for a free account (no credit card required)
3. Navigate to the API Keys section in your dashboard
4. Copy your Trial API key

**Important security practice:** Never hardcode API keys directly in your notebooks or scripts. Store them in a `.env` file instead. This prevents accidentally sharing sensitive credentials when you share your code.

Create a file named `.env` in your project directory with the following entry:

```
COHERE_API_KEY=your_key_here
```

**Important:** Add `.env` to your `.gitignore` file to prevent committing it to version control.

Now load your API key securely:

```python
from dotenv import load_dotenv
import os
from cohere import ClientV2
import time

# Load environment variables from .env file
load_dotenv()

# Access your API key
cohere_api_key = os.getenv('COHERE_API_KEY')

if not cohere_api_key:
    raise ValueError(
        "COHERE_API_KEY not found. Please create a .env file with your API key.\n"
        "See https://dashboard.cohere.com for instructions on getting your key."
    )

# Initialize the Cohere client using the V2 API
co = ClientV2(api_key=cohere_api_key)
print("API key loaded successfully from environment")
```
```
API key loaded successfully from environment
```

Now let's generate embeddings using the Cohere API. Here's something we discovered through trial and error: when we first ran this code without delays, we hit Cohere's rate limit and got a `429 TooManyRequestsError` with this message: "trial token rate limit exceeded, limit is 100000 tokens per minute."

This exposes an important lesson about working with APIs. Rate limits aren't always clearly documented upfront. Sometimes you discover them by running into them, then you have to dig through the error responses in the documentation to understand what happened. In this case, we found the details in Cohere's [error responses documentation](https://docs.cohere.com/v2/reference/errors). You can also check their [rate limits page](https://docs.cohere.com/docs/rate-limits) for current limits, though specifics for free tier accounts aren't always listed there.

With 500 papers averaging around 1,337 characters each, we can easily exceed 100,000 tokens per minute if we send batches too quickly. So we've built in two safeguards: a 12-second delay between batches to stay under the limit, and retry logic in case we do hit it. This makes the process take about 60-70 seconds instead of the 6-8 seconds the API actually needs, but it's reliable and won't throw errors mid-process.

**Think of it as the tradeoff for using a free tier**: we get access to powerful models without paying, but we work within some constraints. Let's see it in action:

```python
print("Generating embeddings using Cohere API...")
print(f"Processing {len(abstracts)} abstracts...")

start_time = time.time()
actual_api_time = 0  # Track time spent on actual API calls

# Cohere recommends processing in batches for efficiency
# Their API accepts up to 96 texts per request
batch_size = 90
all_embeddings = []

for i in range(0, len(abstracts), batch_size):
    batch = abstracts[i:i+batch_size]
    batch_num = i//batch_size + 1
    total_batches = (len(abstracts) + batch_size - 1) // batch_size
    print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} abstracts)...")

    # Add retry logic for rate limits
    max_retries = 3
    retry_delay = 60  # Wait 60 seconds if we hit rate limit

    for attempt in range(max_retries):
        try:
            # Track actual API call time
            api_start = time.time()

            # Generate embeddings for this batch using V2 API
            response = co.embed(
                texts=batch,
                model='embed-v4.0',
                input_type='search_document',
                embedding_types=['float']
            )

            actual_api_time += time.time() - api_start
            # V2 API returns embeddings in a different structure
            all_embeddings.extend(response.embeddings.float_)
            break  # Success, move to next batch

        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                print(f"  Rate limit hit. Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                raise  # Re-raise if it's not a rate limit error or we're out of retries

    # Add a delay between batches to avoid hitting rate limits
    # Wait 12 seconds between batches (spreads 500 papers over ~1 minute)
    if i + batch_size < len(abstracts):  # Don't wait after the last batch
        time.sleep(12)

# Convert to numpy array for consistency with local models
embeddings_cohere = np.array(all_embeddings)
elapsed_time = time.time() - start_time

print(f"\nCompleted in {elapsed_time:.2f} seconds (includes rate limit delays)")
print(f"Actual API processing time: {actual_api_time:.2f} seconds")
print(f"Time spent waiting for rate limits: {elapsed_time - actual_api_time:.2f} seconds")
print(f"Embedding shape: {embeddings_cohere.shape}")
print(f"Each abstract is now a {embeddings_cohere.shape[1]}-dimensional vector")
print(f"Average time per abstract (API only): {actual_api_time/len(abstracts):.3f} seconds")

# Add to DataFrame
df['embedding_cohere'] = list(embeddings_cohere)
```
```
Generating embeddings using Cohere API...
Processing 500 abstracts...
Processing batch 1/6 (90 abstracts)...
Processing batch 2/6 (90 abstracts)...
Processing batch 3/6 (90 abstracts)...
Processing batch 4/6 (90 abstracts)...
Processing batch 5/6 (90 abstracts)...
Processing batch 6/6 (50 abstracts)...

Completed in 87.23 seconds (includes rate limit delays)
Actual API processing time: 27.18 seconds
Time spent waiting for rate limits: 60.05 seconds
Embedding shape: (500, 1536)
Each abstract is now a 1536-dimensional vector
Average time per abstract (API only): 0.054 seconds
```

Notice the timing breakdown? The actual API processing was quite fast (around 27 seconds), but we spent most of our time waiting between batches to respect rate limits (around 60 seconds). This is the reality of free-tier accounts: they're fantastic for learning and prototyping, but come with constraints. Paid tiers would give us much higher limits and let us process at full speed.

**Something else worth noting**: Cohere's embeddings are 1536-dimensional, which is 4x larger than our small local model (384 dimensions) and 2x larger than our large local model (768 dimensions). Yet the API processing was still faster than our small local model. This demonstrates the power of specialized infrastructure. Cohere runs optimized hardware designed specifically for embedding generation at scale, while our local models run on general-purpose computers. Higher dimensions don't automatically mean slower processing when you have the right infrastructure behind them.

For this tutorial, Cohere’s free tier works perfectly. We're focusing on understanding the concepts and comparing approaches, not optimizing for production speed. The key differences from local models:

- **No local computation:** All processing happens on Cohere's servers, so it works equally well on any hardware.
- **Internet dependency:** Requires an active internet connection to work.
- **Rate limits:** Free tier accounts have token-per-minute limits, which is why we added delays between batches.

### Other API Options

While we're using Cohere for this tutorial, you should know about other popular embedding APIs:

**OpenAI** offers excellent embedding models, but requires payment information upfront. If you have an OpenAI account, their `text-embedding-3-small` model is very affordable at $0.02 per 1M tokens. You can find setup instructions in their [embeddings documentation](https://platform.openai.com/docs/guides/embeddings).

**Together AI** provides access to many open-source models through their API. They offer models like `BAAI/bge-large-en-v1.5` and detailed documentation in [their embeddings guide](https://docs.together.ai/docs/embeddings-overview). Note that their rate limit tiers are subject to change, so be sure to check their [rate limit documentation](https://docs.together.ai/docs/rate-limits#rate-limit-tiers) to determine the tier you'll need based on your needs.

The choice between these services depends on your priorities. OpenAI has excellent quality but requires payment setup. Together AI offers many model choices and different paid tiers. Cohere has a truly free tier for learning and prototyping.

## Comparing Your Options

Now that we've generated embeddings using both local models and an API service, let's think about how to choose between these approaches for real projects. The decision isn't about one being universally better than the other. It's about matching the approach to your specific constraints and requirements.

**To clarify terminology**: "self-hosted models" means running models on infrastructure you control, whether that's your laptop for learning or your company's cloud servers for production. "API services" means using third-party providers like Cohere or OpenAI where you send data to their servers for processing.

| Dimension | Self-hosted Models | API Services |
| --- | --- | --- |
| Cost | **Zero ongoing costs** after initial setup. Best for high-volume applications where you'll generate embeddings frequently. | **Pay-per-use model** per 1M tokens. [Cohere](https://www.dataquest.io/blog/generating-embeddings-with-apis-and-open-models/%E2%80%9Dhttps://cohere.com/pricing%E2%80%9D): $0.12 per 1M tokens. [OpenAI](https://www.dataquest.io/blog/generating-embeddings-with-apis-and-open-models/%E2%80%9Dhttps://platform.openai.com/docs/pricing#embeddings%E2%80%9D): $0.13 per 1M tokens. Best for low to moderate volume, or when you want predictable costs without infrastructure. |
| Performance | **Speed depends on your hardware.** Our results: 0.131 seconds per abstract (small model), 1.361 seconds per abstract (large model). Best for batch processing or when you control the infrastructure. | **Speed depends on internet connection** and API server load. Our results: 0.054 seconds per abstract (Cohere). Includes network latency and third-party infrastructure considerations. Best when you don't have powerful local hardware or need access to the latest models. |
| Privacy | **All data stays on your infrastructure.** Complete control over data handling. No data sent to third parties. Best for sensitive data, healthcare, financial services, or when compliance requires data locality. | **Data is sent to third-party servers** for processing. Subject to the provider's data handling policies. Cohere states that API data isn't used for training (verify current policy). Best for non-sensitive data, or when provider policies meet your requirements. |
| Customization | **Can fine-tune models** on your specific domain. Full control over model selection and updates. Can modify inference parameters. Best for specialized domains, custom requirements, or when you need reproducibility. | **Limited to provider's available models.** Model updates happen on provider's schedule. Less control over inference details. Best for general-purpose applications, or when using the latest models matters more than control. |
| Infrastructure | **Requires managing infrastructure.** Whether running on your laptop or company cloud servers, you handle model updates, dependencies, and scaling. Best for organizations with existing ML infrastructure or when infrastructure control is important. | **No infrastructure management needed.** Automatic scaling to handle load. Provider manages updates and availability. Best for smaller teams, rapid prototyping, or when you want to focus on application logic rather than infrastructure. |

### When to Use Each Approach

Here's a practical decision guide to help you choose the right approach for your project:

> Choose Self-Hosted Models when you:
> 
> - Process large volumes of text regularly
> - Work with sensitive or regulated data
> - Need offline capability
> - Have existing ML infrastructure (whether local or cloud-based)
> - Want to fine-tune models for your domain
> - Need complete control over the deployment
> 
> Choose API Services when you:
> 
> - Are just getting started or prototyping
> - Have unpredictable or variable workload
> - Want to avoid infrastructure management
> - Need automatic scaling
> - Prefer the latest models without maintenance
> - Value simplicity over infrastructure control

For our tutorial series, we've used both approaches to give you hands-on experience with each. In our next tutorial, we'll use the Cohere embeddings for our semantic search implementation. We're choosing Cohere because they offer a generous free tier for learning (no payment required), their models are well-suited for semantic search tasks, and they work consistently across different hardware setups.

In practice, you'd evaluate embedding quality by testing on your specific use case: generate embeddings with different models, run similarity searches on sample queries, and measure which model returns the most relevant results for your domain.

## Storing Your Embeddings

We've generated embeddings using multiple methods, and now we need to save them for future use. Storing embeddings properly is important because generating them can be time-consuming and potentially costly. You don't want to regenerate embeddings every time you run your code.

Let's explore two storage approaches:

### Option 1: CSV with Numpy Arrays

This approach works well for learning and small-scale prototyping:

```python
# Save the metadata to CSV (without embeddings, which are large arrays)
df_metadata = df[['title', 'abstract', 'authors', 'published', 'category', 'arxiv_id', 'abstract_length']]
df_metadata.to_csv('arxiv_papers_metadata.csv', index=False)
print("Saved metadata to 'arxiv_papers_metadata.csv'")

# Save embeddings as numpy arrays
np.save('embeddings_minilm.npy', embeddings_small)
np.save('embeddings_mpnet.npy', embeddings_large)
np.save('embeddings_cohere.npy', embeddings_cohere)
print("Saved embeddings to .npy files")

# Later, you can load them back like this:
# df_loaded = pd.read_csv('arxiv_papers_metadata.csv')
# embeddings_loaded = np.load('embeddings_cohere.npy')
```
```
Saved metadata to 'arxiv_papers_metadata.csv'
Saved embeddings to .npy files
```

This approach is simple and transparent, making it perfect for learning and experimentation. However, it has significant limitations for larger datasets:

- Loading all embeddings into memory doesn't scale beyond a few thousand documents
- No indexing for fast similarity search
- Manual coordination between CSV metadata and numpy arrays

For production systems with thousands or millions of embeddings, you'll want specialized vector databases (Option 2) that handle indexing, similarity search, and efficient storage automatically.

### Option 2: Preparing for Vector Databases

In production systems, you'll likely store embeddings in a specialized vector database like Pinecone, Weaviate, or Chroma. These databases are optimized for similarity search. While we'll cover vector databases in detail in another tutorial series, here's how you'd structure your data for them:

```python
# Prepare data in a format suitable for vector databases
# Most vector databases want: ID, embedding vector, and metadata

vector_db_data = []
for idx, row in df.iterrows():
    vector_db_data.append({
        'id': row['arxiv_id'],
        'embedding': row['embedding_cohere'].tolist(),  # Convert numpy array to list
        'metadata': {
            'title': row['title'],
            'abstract': row['abstract'][:500],  # Many DBs limit metadata size
            'authors': ', '.join(row['authors'][:3]),  # First 3 authors
            'category': row['category'],
            'published': str(row['published'])
        }
    })

# Save in JSON format for easy loading into vector databases
import json
with open('arxiv_papers_vector_db_format.json', 'w') as f:
    json.dump(vector_db_data, f, indent=2)
print("Saved data in vector database format to 'arxiv_papers_vector_db_format.json'")

print(f"\nTotal storage sizes:")
print(f"  Metadata CSV: ~{os.path.getsize('arxiv_papers_metadata.csv')/1024:.1f} KB")
print(f"  JSON for vector DB: ~{os.path.getsize('arxiv_papers_vector_db_format.json')/1024:.1f} KB")
```
```
Saved data in vector database format to 'arxiv_papers_vector_db_format.json'

Total storage sizes:
  Metadata CSV: ~764.6 KB
  JSON for vector DB: ~15051.0 KB
```

Each storage method has its purpose:

- **CSV + numpy**: Best for learning and small-scale experimentation
- **JSON for vector databases**: Best for production systems that need efficient similarity search

## Preparing for Semantic Search

You now have 500 research papers from five distinct computer science domains with embeddings that capture their semantic meaning. **These embeddings are vectors, which means we can measure how similar or different they are using mathematical distance calculations.**

In the next tutorial, you'll use these embeddings to build a search system that finds relevant papers based on meaning rather than keywords. You'll implement similarity calculations, rank results, and see firsthand how semantic search outperforms traditional keyword matching.

Save your embeddings now, especially the Cohere embeddings since we'll use those in the next tutorial to build our search system. We chose Cohere because they work consistently across different hardware setups and provide a consistent baseline for implementing similarity calculations.

## Next Steps

Before we move on, try these experiments to deepen your understanding:

**Experiment with different arXiv categories:**

- Try collecting papers from categories like `stat.ML` (Statistics Machine Learning) or `math.OC` (Optimization and Control)
- Use the PCA visualization code to see how these new categories cluster with your existing five
- Do some categories overlap more than others?

**Compare embedding models visually:**

- Generate embeddings for your dataset using `all-mpnet-base-v2`
- Create side-by-side PCA visualizations comparing the small model and large model
- Do the clusters look tighter or more separated with the larger model?

**Test different dataset sizes:**

- Collect just 50 papers per category (250 total) and visualize the results
- Then try 200 papers per category (1000 total)
- How does dataset size affect the clarity of the clusters?
- At what point does collection or processing time become noticeable?

**Explore model differences:**

- Visit [Hugging Face's sentence similarity models](https://huggingface.co/models?pipeline_tag=sentence-similarity)
- Try a domain-specific model optimized for scientific text
- Compare the embeddings qualitatively by looking at which papers cluster together

Ready to implement similarity search and build a working semantic search engine? The next tutorial will show you how to turn these embeddings into a powerful research discovery tool.

---

**Key Takeaways**:

- Programmatic data collection through APIs like arXiv enables working with real-world datasets
- Collecting papers from diverse categories (cs.LG, cs.CV, cs.CL, cs.DB, cs.SE) creates semantic clusters for effective search
- Papers can be cross-listed in multiple arXiv categories, creating natural overlap between related fields
- Self-hosted embedding models provide zero-cost, private embedding generation with full control over the process
- API-based embedding services offer high-quality embeddings without infrastructure management
- Secure credential handling using `.env` files protects sensitive API keys and tokens
- Rate limits aren't always clearly documented and are sometimes discovered through trial and error
- The choice between self-hosted and API approaches depends on cost, privacy, scale, and infrastructure considerations
- Free tier APIs provide powerful embedding generation for learning, but require handling rate limits and delays that paid tiers avoid
- Real-world embeddings show more overlap than handcrafted examples, reflecting the complexity of actual data
- Proper storage of embeddings prevents costly regeneration and enables efficient reuse across projects

# 3. Measuring Similarity and Distance between Embeddings

[***Taken From***](https://www.dataquest.io/blog/measuring-similarity-and-distance-between-embeddings/)

In the previous tutorial, you learned how to collect 500 research papers from arXiv and generate embeddings using both local models and API services. You now have a dataset of papers with embeddings that capture their semantic meaning. Those embeddings are vectors, which means we can perform mathematical operations on them.

But here's the thing: **having embeddings isn't enough to build a search system**. You need to know how to measure similarity between vectors. When a user searches for "neural networks for computer vision," which papers in your dataset are most relevant? The answer lies in **measuring the distance between the query embedding and each paper embedding**.

This tutorial teaches you how to implement similarity calculations and build a functional semantic search engine. You'll implement three different distance metrics, understand when to use each one, and create a search function that returns ranked results based on semantic similarity. By the end, you'll have built a complete search system that finds relevant papers based on meaning rather than keywords.

## Setting Up Your Environment

We'll continue using the same libraries from the previous tutorials. If you've been following along, you should already have these installed. If not, here's the installation command for you to run from your terminal:

```bash
# Developed with: Python 3.12.12
# scikit-learn==1.6.1
# matplotlib==3.10.0
# numpy==2.0.2
# pandas==2.2.2
# cohere==5.20.0
# python-dotenv==1.1.1

pip install scikit-learn matplotlib numpy pandas cohere python-dotenv
```

## Loading Your Saved Embeddings

Previously, we saved our embeddings and metadata (`arxiv_papers_metadata.csv`) to disk. Let's load them back into memory so we can work with them. We'll use the Cohere embeddings (`embeddings_cohere.npy`) because they provide consistent results across different hardware setups. If you don't have these files, you can download them [here](https://dq-content.s3.us-east-1.amazonaws.com/3000045/arxiv_metadata_and_embeddings.zip).

```python
import numpy as np
import pandas as pd

# Load the metadata
df = pd.read_csv('arxiv_papers_metadata.csv')
print(f"Loaded {len(df)} papers")

# Load the Cohere embeddings
embeddings = np.load('embeddings_cohere.npy')
print(f"Loaded embeddings with shape: {embeddings.shape}")
print(f"Each paper is represented by a {embeddings.shape[1]}-dimensional vector")

# Verify the data loaded correctly
print(f"\nFirst paper title: {df['title'].iloc[0]}")
print(f"First embedding (first 5 values): {embeddings[0][:5]}")
```
```
Loaded 500 papers
Loaded embeddings with shape: (500, 1536)
Each paper is represented by a 1536-dimensional vector

First paper title: Dark Energy Survey Year 3 results: Simulation-based $w$CDM inference from weak lensing and galaxy clustering maps with deep learning. I. Analysis design
First embedding (first 5 values): [-7.7144260e-03  1.9527141e-02 -4.2141182e-05 -2.8627755e-03 -2.5192423e-02]
```

Perfect! We have our 500 papers and their corresponding 1536-dimensional embedding vectors. Each vector is a point in high-dimensional space, and papers with similar content will have vectors that are close together. Now we need to define what "close together" actually means.

## Understanding Distance in Vector Space

Before we write any code, let's build intuition about measuring similarity between vectors. Imagine you have two papers about software compliance. Their embeddings might look like this:

```
Paper A: [0.8, 0.6, 0.1, ...]  (1536 numbers total)
Paper B: [0.7, 0.5, 0.2, ...]  (1536 numbers total)
```

To calculate the distance between embedding vectors, we need a distance metric. There are three commonly used metrics for measuring similarity between embeddings:

1. **Euclidean Distance**: Measures the straight-line distance between vectors in space. A shorter distance means higher similarity. You can think of it as measuring the physical distance between two points.
2. **Dot Product**: Multiplies corresponding elements and sums them up. Considers both direction and magnitude of the vectors. Works well when embeddings are normalized to unit length.
3. **Cosine Similarity**: Measures the angle between vectors. If two vectors point in the same direction, they're similar, regardless of their length. This is the most common metric for text embeddings.

We'll implement each metric in order from most intuitive to most commonly used. Let's start with Euclidean distance because it's the easiest to understand.

## Implementing Euclidean Distance

Euclidean distance measures the straight-line distance between two points in space. This is the most intuitive metric because we all understand physical distance. If you have two points on a map, the Euclidean distance is literally how far apart they are.

Unlike the other metrics we'll learn (where higher is better), Euclidean distance works in reverse: lower distance means higher similarity. Papers that are close together in space have low distance and are semantically similar.

Note that Euclidean distance is sensitive to vector magnitude. If your embeddings aren't normalized (meaning vectors can have different lengths), two vectors pointing in similar directions but with different magnitudes will show larger distance than expected. This is why cosine similarity (which we'll learn next) is often preferred for text embeddings. It ignores magnitude and focuses purely on direction.

The formula is:

$\text{Euclidean distance} = \left|\right. \mathbf{A} - \mathbf{B} \left|\right. = \sqrt{\sum_{i = 1}^{n} \left(\right. A_{i} - B_{i} \left.\right)^{2}}$ 
$$
\text{Euclidean distance} = |\mathbf{A} - \mathbf{B}| = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
$$

This is essentially the Pythagorean theorem extended to high-dimensional space. We subtract corresponding values, square them, sum everything up, and take the square root. Let's implement it:

```python
def euclidean_distance_manual(vec1, vec2):
    """
    Calculate Euclidean distance between two vectors.

    Parameters:
    -----------
    vec1, vec2 : numpy arrays
        The vectors to compare

    Returns:
    --------
    float
        Euclidean distance (lower means more similar)
    """
    # np.linalg.norm computes the square root of sum of squared differences
    # This implements the Euclidean distance formula directly
    return np.linalg.norm(vec1 - vec2)

# Let's test it by comparing two similar papers
paper_idx_1 = 492  # Android compliance detection paper
paper_idx_2 = 493  # GDPR benchmarking paper

distance = euclidean_distance_manual(embeddings[paper_idx_1], embeddings[paper_idx_2])

print(f"Comparing two papers:")
print(f"Paper 1: {df['title'].iloc[paper_idx_1][:50]}...")
print(f"Paper 2: {df['title'].iloc[paper_idx_2][:50]}...")
print(f"\nEuclidean distance: {distance:.4f}")
```
```
Comparing two papers:
Paper 1: Can Large Language Models Detect Real-World Androi...
Paper 2: GDPR-Bench-Android: A Benchmark for Evaluating Aut...

Euclidean distance: 0.8431
```

A distance of 0.84 is quite low, which means these papers are very similar! Both papers discuss Android compliance and benchmarking, so this makes perfect sense. Now let's compare this to a paper from a completely different category:

```python
# Compare a software engineering paper to a database paper
paper_idx_3 = 300  # A database paper about natural language queries

distance_related = euclidean_distance_manual(embeddings[paper_idx_1],
                                             embeddings[paper_idx_2])
distance_unrelated = euclidean_distance_manual(embeddings[paper_idx_1],
                                               embeddings[paper_idx_3])

print(f"Software Engineering paper 1 vs Software Engineering paper 2:")
print(f"  Distance: {distance_related:.4f}")
print(f"\nSoftware Engineering paper vs Database paper:")
print(f"  Distance: {distance_unrelated:.4f}")
print(f"\nThe related SE papers are {distance_unrelated/distance_related:.2f}x closer")
```
```
Software Engineering paper 1 vs Software Engineering paper 2:
  Distance: 0.8431

Software Engineering paper vs Database paper:
  Distance: 1.2538

The related SE papers are 1.49x closer
```

The distance correctly identifies that papers from the same category are closer to each other than papers from different categories. The related papers have a much lower distance.

For calculating distance to all papers, we can use scikit-learn:

```python
from sklearn.metrics.pairwise import euclidean_distances

# Calculate distance from one paper to all others
query_embedding = embeddings[paper_idx_1].reshape(1, -1)
all_distances = euclidean_distances(query_embedding, embeddings)

# Get top 10 (lowest distances = most similar)
top_indices = np.argsort(all_distances[0])[1:11]

print(f"Query paper: {df['title'].iloc[paper_idx_1]}\n")
print("Top 10 papers by Euclidean distance (lowest = most similar):")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. [{all_distances[0][idx]:.4f}] {df['title'].iloc[idx][:50]}...")
```
```
Query paper: Can Large Language Models Detect Real-World Android Software Compliance Violations?

Top 10 papers by Euclidean distance (lowest = most similar):
1. [0.8431] GDPR-Bench-Android: A Benchmark for Evaluating Aut...
2. [1.0168] An Empirical Study of LLM-Based Code Clone Detecti...
3. [1.0218] LLM-as-a-Judge is Bad, Based on AI Attempting the ...
4. [1.0541] BengaliMoralBench: A Benchmark for Auditing Moral ...
5. [1.0677] Exploring the Feasibility of End-to-End Large Lang...
6. [1.0730] Where Do LLMs Still Struggle? An In-Depth Analysis...
7. [1.0730] Where Do LLMs Still Struggle? An In-Depth Analysis...
8. [1.0763] EvoDev: An Iterative Feature-Driven Framework for ...
9. [1.0766] Watermarking Large Language Models in Europe: Inte...
10. [1.0814] One Battle After Another: Probing LLMs' Limits on ...
```

Euclidean distance is intuitive and works well for many applications. Now let's learn about dot product, which takes a different approach to measuring similarity.

## Implementing Dot Product Similarity

The dot product is simpler than Euclidean distance because it doesn't involve taking square roots or differences. Instead, we multiply corresponding elements and sum them up. The formula is:

$\text{dot product} = \mathbf{A} \cdot \mathbf{B} = \sum_{i = 1}^{n} A_{i} B_{i}$ 
$$
\text{dot product} = \mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} A_i B_i
$$

The dot product considers both the angle between vectors and their magnitudes. When vectors point in similar directions, the products of corresponding elements tend to be positive and large, resulting in a high dot product. When vectors point in different directions, some products are positive and some negative, and they tend to cancel out, resulting in a lower dot product. Higher scores mean higher similarity.

The dot product works particularly well when embeddings have been normalized to similar lengths. Many embedding APIs like Cohere and OpenAI produce normalized embeddings by default. However, some open-source frameworks (like sentence-transformers or instructor) require you to explicitly set normalization parameters. Always check your embedding model's documentation to understand whether normalization is applied automatically or needs to be configured.

Let's implement it:

```python
def dot_product_similarity_manual(vec1, vec2):
    """
    Calculate dot product between two vectors.

    Parameters:
    -----------
    vec1, vec2 : numpy arrays
        The vectors to compare

    Returns:
    --------
    float
        Dot product score (higher means more similar)
    """
    # np.dot multiplies corresponding elements and sums them
    # This directly implements the dot product formula
    return np.dot(vec1, vec2)

# Compare the same papers using dot product
similarity_dot = dot_product_similarity_manual(embeddings[paper_idx_1],
                                               embeddings[paper_idx_2])

print(f"Comparing the same two papers:")
print(f"  Dot product: {similarity_dot:.4f}")
```
```
Comparing the same two papers:
  Dot product: 0.6446
```

Keep this number in mind. When we calculate cosine similarity next, you'll see why dot product works so well for these embeddings.

For search across all papers, we can use NumPy's matrix multiplication:

```python
# Efficient dot product for one query against all papers
query_embedding = embeddings[paper_idx_1]
all_dot_products = np.dot(embeddings, query_embedding)

# Get top 10 results
top_indices = np.argsort(all_dot_products)[::-1][1:11]

print(f"Query paper: {df['title'].iloc[paper_idx_1]}\n")
print("Top 10 papers by dot product similarity:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. [{all_dot_products[idx]:.4f}] {df['title'].iloc[idx][:50]}...")
```
```
Query paper: Can Large Language Models Detect Real-World Android Software Compliance Violations?

Top 10 papers by dot product similarity:
1. [0.6446] GDPR-Bench-Android: A Benchmark for Evaluating Aut...
2. [0.4831] An Empirical Study of LLM-Based Code Clone Detecti...
3. [0.4779] LLM-as-a-Judge is Bad, Based on AI Attempting the ...
4. [0.4445] BengaliMoralBench: A Benchmark for Auditing Moral ...
5. [0.4300] Exploring the Feasibility of End-to-End Large Lang...
6. [0.4243] Where Do LLMs Still Struggle? An In-Depth Analysis...
7. [0.4243] Where Do LLMs Still Struggle? An In-Depth Analysis...
8. [0.4208] EvoDev: An Iterative Feature-Driven Framework for ...
9. [0.4204] Watermarking Large Language Models in Europe: Inte...
10. [0.4153] One Battle After Another: Probing LLMs' Limits on ...
```

Notice that the rankings are identical to those from Euclidean distance! This happens because both metrics capture similar relationships in the data, just measured differently. This won't always be the case with all embedding models, but it's common when embeddings are well-normalized.

## Implementing Cosine Similarity

Cosine similarity is the most commonly used metric for text embeddings. It measures the angle between vectors rather than their distance. If two vectors point in the same direction, they're similar, regardless of how long they are.

The formula looks like this:

$\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\left|\right. \mathbf{A} \left|\right. \left|\right. \mathbf{B} \left|\right.} = \frac{\sum_{i = 1}^{n} A_{i} B_{i}}{\sqrt{\sum_{i = 1}^{n} A_{i}^{2}} \sqrt{\sum_{i = 1}^{n} B_{i}^{2}}}$ 
$$
\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}| |\mathbf{B}|}=\frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

Where $\mathbf{A}$ and $\mathbf{B}$ are our two vectors, $\mathbf{A} \cdot \mathbf{B}$ is the dot product, and $\left|\right. \mathbf{A} \left|\right.$ represents the magnitude (or length) of vector $\mathbf{A}$.

The result ranges from -1 to 1:

- **1** means the vectors point in exactly the same direction (identical meaning)
- **0** means the vectors are perpendicular (unrelated)
- **\-1** means the vectors point in opposite directions (opposite meaning)

For text embeddings, you'll typically see values between 0 and 1 because embeddings rarely point in completely opposite directions.

Let's implement this using NumPy:

```python
def cosine_similarity_manual(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.

    Parameters:
    -----------
    vec1, vec2 : numpy arrays
        The vectors to compare

    Returns:
    --------
    float
        Cosine similarity score between -1 and 1
    """
    # Calculate dot product (numerator)
    dot_product = np.dot(vec1, vec2)

    # Calculate magnitudes using np.linalg.norm (denominator)
    # np.linalg.norm computes sqrt(sum of squared values)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Divide dot product by product of magnitudes
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity

# Test with our software engineering papers
similarity = cosine_similarity_manual(embeddings[paper_idx_1],
                                     embeddings[paper_idx_2])

print(f"Comparing two papers:")
print(f"Paper 1: {df['title'].iloc[paper_idx_1][:50]}...")
print(f"Paper 2: {df['title'].iloc[paper_idx_2][:50]}...")
print(f"\nCosine similarity: {similarity:.4f}")
```
```
Comparing two papers:
Paper 1: Can Large Language Models Detect Real-World Androi...
Paper 2: GDPR-Bench-Android: A Benchmark for Evaluating Aut...

Cosine similarity: 0.6446
```

The cosine similarity (0.6446) is identical to the dot product we calculated earlier. This isn't a coincidence. Cohere's embeddings are normalized to unit length, which means the dot product and cosine similarity are mathematically equivalent for these vectors. When embeddings are normalized, the denominator in the cosine formula (the product of the vector magnitudes) always equals 1, leaving just the dot product. This is why many vector databases prefer dot product for normalized embeddings. It's computationally cheaper and produces identical results to cosine.

Now let's compare this to a paper from a completely different category:

```python
# Compare a software engineering paper to a database paper
similarity_related = cosine_similarity_manual(embeddings[paper_idx_1],
                                             embeddings[paper_idx_2])
similarity_unrelated = cosine_similarity_manual(embeddings[paper_idx_1],
                                               embeddings[paper_idx_3])

print(f"Software Engineering paper 1 vs Software Engineering paper 2:")
print(f"  Similarity: {similarity_related:.4f}")
print(f"\nSoftware Engineering paper vs Database paper:")
print(f"  Similarity: {similarity_unrelated:.4f}")
print(f"\nThe SE papers are {similarity_related/similarity_unrelated:.2f}x more similar")
```
```
Software Engineering paper 1 vs Software Engineering paper 2:
  Similarity: 0.6446

Software Engineering paper vs Database paper:
  Similarity: 0.2140

The SE papers are 3.01x more similar
```

Great! The similarity score correctly identifies that papers from the same category are much more similar to each other than papers from different categories.

Now, calculating similarity one pair at a time is fine for understanding, but it's not practical for search. We need to compare a query against all 500 papers efficiently. Let's use scikit-learn's optimized implementation:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between one paper and all other papers
query_embedding = embeddings[paper_idx_1].reshape(1, -1)
all_similarities = cosine_similarity(query_embedding, embeddings)

# Get the top 10 most similar papers (excluding the query itself)
top_indices = np.argsort(all_similarities[0])[::-1][1:11]

print(f"Query paper: {df['title'].iloc[paper_idx_1]}\n")
print("Top 10 most similar papers:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. [{all_similarities[0][idx]:.4f}] {df['title'].iloc[idx][:50]}...")
```
```
Query paper: Can Large Language Models Detect Real-World Android Software Compliance Violations?

Top 10 most similar papers:
1. [0.6446] GDPR-Bench-Android: A Benchmark for Evaluating Aut...
2. [0.4831] An Empirical Study of LLM-Based Code Clone Detecti...
3. [0.4779] LLM-as-a-Judge is Bad, Based on AI Attempting the ...
4. [0.4445] BengaliMoralBench: A Benchmark for Auditing Moral ...
5. [0.4300] Exploring the Feasibility of End-to-End Large Lang...
6. [0.4243] Where Do LLMs Still Struggle? An In-Depth Analysis...
7. [0.4243] Where Do LLMs Still Struggle? An In-Depth Analysis...
8. [0.4208] EvoDev: An Iterative Feature-Driven Framework for ...
9. [0.4204] Watermarking Large Language Models in Europe: Inte...
10. [0.4153] One Battle After Another: Probing LLMs' Limits on ...
```

Notice how scikit-learn's `cosine_similarity` function is much cleaner. It handles the reshaping and broadcasts the calculation efficiently across all papers. This is what you'll use in production code, but understanding the manual implementation helps you see what's happening under the hood.

You might notice papers 6 and 7 appear to be duplicates with identical scores. This happens because the same paper was cross-listed in multiple arXiv categories. In a production system, you'd typically de-duplicate results using a stable identifier like the arXiv ID, showing each unique paper only once while perhaps noting all its categories.

## Choosing the Right Metric for Your Use Case

Now that we've implemented all three metrics, let's understand when to use each one. Here's a practical comparison:

| Metric | When to Use | Advantages | Considerations |
| --- | --- | --- | --- |
| **Euclidean Distance** | Use when the absolute position in vector space matters, or for scientific computing applications. | Intuitive geometric interpretation. Common in general machine learning tasks beyond NLP. | Lower scores mean higher similarity (inverse relationship). Can be sensitive to vector magnitude. |
| **Dot Product** | Use when embeddings are already normalized to unit length. Common in vector databases. | Fastest computation. Identical rankings to cosine for normalized vectors. Many vector DBs optimize for this. | Only equivalent to cosine when vectors are normalized. Check your embedding model's documentation. |
| **Cosine Similarity** | Default choice for text embeddings. Use when you care about semantic similarity regardless of document length. | Most common in NLP. Normalized by default (outputs 0 to 1). Works well with sentence-transformers and most embedding APIs. | Requires normalization calculation. Slightly more computationally expensive than dot product. |

Going forward, we'll use cosine similarity because it's the standard for text embeddings and produces interpretable scores between 0 and 1.

Let's verify that our embeddings produce consistent rankings across metrics:

```python
# Compare rankings from all three metrics for a single query
query_embedding = embeddings[paper_idx_1].reshape(1, -1)

# Calculate similarities/distances
cosine_scores = cosine_similarity(query_embedding, embeddings)[0]
dot_scores = np.dot(embeddings, embeddings[paper_idx_1])
euclidean_scores = euclidean_distances(query_embedding, embeddings)[0]

# Get top 10 indices for each metric
top_cosine = set(np.argsort(cosine_scores)[::-1][1:11])
top_dot = set(np.argsort(dot_scores)[::-1][1:11])
top_euclidean = set(np.argsort(euclidean_scores)[1:11])

# Calculate overlap
cosine_dot_overlap = len(top_cosine & top_dot)
cosine_euclidean_overlap = len(top_cosine & top_euclidean)
all_three_overlap = len(top_cosine & top_dot & top_euclidean)

print(f"Top 10 papers overlap between metrics:")
print(f"  Cosine & Dot Product: {cosine_dot_overlap}/10 papers match")
print(f"  Cosine & Euclidean: {cosine_euclidean_overlap}/10 papers match")
print(f"  All three metrics: {all_three_overlap}/10 papers match")
```
```
Top 10 papers overlap between metrics:
  Cosine & Dot Product: 10/10 papers match
  Cosine & Euclidean: 10/10 papers match
  All three metrics: 10/10 papers match
```

For our Cohere embeddings with these 500 papers, all three metrics produce identical top-10 rankings. This happens when embeddings are well-normalized, but isn't guaranteed across all embedding models or datasets. What matters more than perfect metric agreement is understanding what each metric measures and when to use it.

## Building Your Search Function

Now let's build a complete semantic search function that ties everything together. This function will take a natural language query, convert it into an embedding, and return the most relevant papers.

Before building our search function, ensure your Cohere API key is configured. As we did in the previous tutorial, you should have a `.env` file in your project directory with your API key:

```
COHERE_API_KEY=your_key_here
```

Now let's build the search function:

```python
from cohere import ClientV2
from dotenv import load_dotenv
import os

# Load Cohere API key
load_dotenv()
cohere_api_key = os.getenv('COHERE_API_KEY')
co = ClientV2(api_key=cohere_api_key)

def semantic_search(query, embeddings, df, top_k=5, metric='cosine'):
    """
    Search for papers semantically similar to a query.

    Parameters:
    -----------
    query : str
        Natural language search query
    embeddings : numpy array
        Pre-computed embeddings for all papers
    df : pandas DataFrame
        DataFrame containing paper metadata
    top_k : int
        Number of results to return
    metric : str
        Similarity metric to use ('cosine', 'dot', or 'euclidean')

    Returns:
    --------
    pandas DataFrame
        Top results with similarity scores
    """
    # Generate embedding for the query
    response = co.embed(
        texts=[query],
        model='embed-v4.0',
        input_type='search_query',
        embedding_types=['float']
    )
    query_embedding = np.array(response.embeddings.float_[0]).reshape(1, -1)

    # Calculate similarities based on chosen metric
    if metric == 'cosine':
        scores = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
    elif metric == 'dot':
        scores = np.dot(embeddings, query_embedding.flatten())
        top_indices = np.argsort(scores)[::-1][:top_k]
    elif metric == 'euclidean':
        scores = euclidean_distances(query_embedding, embeddings)[0]
        top_indices = np.argsort(scores)[:top_k]
        scores = 1 / (1 + scores)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Create results DataFrame
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = scores[top_indices]
    results = results[['title', 'category', 'similarity_score', 'abstract']]

    return results

# Test the search function
query = "query optimization algorithms"
results = semantic_search(query, embeddings, df, top_k=5)

separator = "=" * 80
print(f"Query: '{query}'\n")
print(f"Top 5 most relevant papers:\n{separator}")
for idx, row in results.iterrows():
    print(f"\n{row['title']}")
    print(f"Category: {row['category']} | Similarity: {row['similarity_score']:.4f}")
    print(f"Abstract: {row['abstract'][:150]}...")
```
```
Query: 'query optimization algorithms'

Top 5 most relevant papers:
================================================================================

Query Optimization in the Wild: Realities and Trends
Category: cs.DB | Similarity: 0.4206
Abstract: For nearly half a century, the core design of query optimizers in industrial database systems has remained remarkably stable, relying on foundational
...

Hybrid Mixed Integer Linear Programming for Large-Scale Join Order Optimisation
Category: cs.DB | Similarity: 0.3795
Abstract: Finding optimal join orders is among the most crucial steps to be performed by query optimisers. Though extensively studied in data management researc...

One Join Order Does Not Fit All: Reducing Intermediate Results with Per-Split Query Plans
Category: cs.DB | Similarity: 0.3682
Abstract: Minimizing intermediate results is critical for efficient multi-join query processing. Although the seminal Yannakakis algorithm offers strong guarant...

PathFinder: Efficiently Supporting Conjunctions and Disjunctions for Filtered Approximate Nearest Neighbor Search
Category: cs.DB | Similarity: 0.3673
Abstract: Filtered approximate nearest neighbor search (ANNS) restricts the search to data objects whose attributes satisfy a given filter and retrieves the top...

Fine-Grained Dichotomies for Conjunctive Queries with Minimum or Maximum
Category: cs.DB | Similarity: 0.3666
Abstract: We investigate the fine-grained complexity of direct access to Conjunctive Query (CQ) answers according to their position, ordered by the minimum (or
...
```

Excellent! Our search function found highly relevant papers about query optimization. Notice how all the top results are from the `cs.DB` (Databases) category and have strong similarity scores.

Before we move on, let's talk about what these similarity scores mean. Notice our top score is around 0.42 rather than 0.85 or higher. This is completely normal for multi-domain datasets. We're working with 500 papers spanning five distinct computer science fields (Machine Learning, Computer Vision, NLP, Databases, Software Engineering). When your dataset covers diverse topics, even genuinely relevant papers show moderate absolute scores because the overall vocabulary space is broad.

If we had a specialized dataset focused narrowly on one topic, say only database query optimization papers, we'd see higher absolute scores. What matters most is relative ranking. The top results are still the most relevant papers, and the ranking accurately reflects semantic similarity. Pay attention to the score differences between results rather than treating specific thresholds as universal truths.

Let's test it with a few more diverse queries to see how well it works across different topics:

```python
# Test multiple queries
test_queries = [
    "language model pretraining",
    "reinforcement learning algorithms",
    "code quality analysis"
]

for query in test_queries:
    print(f"\nQuery: '{query}'\n{separator}")
    results = semantic_search(query, embeddings, df, top_k=3)

    for idx, row in results.iterrows():
        print(f"  [{row['similarity_score']:.4f}] {row['title'][:50]}...")
        print(f"           Category: {row['category']}")
```
```
Query: 'language model pretraining'
================================================================================
  [0.4240] Reusing Pre-Training Data at Test Time is a Comput...
           Category: cs.CL
  [0.4102] Evo-1: Lightweight Vision-Language-Action Model wi...
           Category: cs.CV
  [0.3910] PixCLIP: Achieving Fine-grained Visual Language Un...
           Category: cs.CV

Query: 'reinforcement learning algorithms'
================================================================================
  [0.3477] Exchange Policy Optimization Algorithm for Semi-In...
           Category: cs.LG
  [0.3429] Fitting Reinforcement Learning Model to Behavioral...
           Category: cs.LG
  [0.3091] Online Algorithms for Repeated Optimal Stopping: A...
           Category: cs.LG

Query: 'code quality analysis'
================================================================================
  [0.3762] From Code Changes to Quality Gains: An Empirical S...
           Category: cs.SE
  [0.3662] Speed at the Cost of Quality? The Impact of LLM Ag...
           Category: cs.SE
  [0.3502] A Systematic Literature Review of Code Hallucinati...
           Category: cs.SE
```

The search function correctly identifies relevant papers for each query. The language model query returns papers about training language models. The reinforcement learning query returns papers about RL algorithms. The code quality query returns papers about testing and technical debt.

Notice how the semantic search understands the meaning behind the queries, not just keyword matching. Even though our queries use natural language, the system finds papers that match the intent.

## Visualizing Search Results in Embedding Space

We've seen the search function work, but let's visualize what's actually happening in embedding space. This will help you understand why certain papers are retrieved for a given query. We'll use PCA to reduce our embeddings to 2D and show how the query relates spatially to its results:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_search_results(query, embeddings, df, top_k=10):
    """
    Visualize search results in 2D embedding space.
    """
    # Get search results
    response = co.embed(
        texts=[query],
        model='embed-v4.0',
        input_type='search_query',
        embedding_types=['float']
    )
    query_embedding = np.array(response.embeddings.float_[0])

    # Calculate similarities
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Combine query embedding with all paper embeddings for PCA
    all_embeddings_with_query = np.vstack([query_embedding, embeddings])

    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings_with_query)

    # Split back into query and papers
    query_2d = embeddings_2d[0]
    papers_2d = embeddings_2d[1:]

    # Create visualization
    plt.figure(figsize=(8, 6))

    # Define colors for categories
    colors = ['#C8102E', '#003DA5', '#00843D', '#FF8200', '#6A1B9A']
    category_codes = ['cs.LG', 'cs.CV', 'cs.CL', 'cs.DB', 'cs.SE']
    category_names = ['Machine Learning', 'Computer Vision', 'Comp. Linguistics',
                     'Databases', 'Software Eng.']

    # Plot all papers with subtle colors
    for i, (cat_code, cat_name, color) in enumerate(zip(category_codes,
                                                         category_names, colors)):
        mask = df['category'] == cat_code
        cat_embeddings = papers_2d[mask]
        plt.scatter(cat_embeddings[:, 0], cat_embeddings[:, 1],
                   c=color, label=cat_name, s=30, alpha=0.3, edgecolors='none')

    # Highlight top results
    top_embeddings = papers_2d[top_indices]
    plt.scatter(top_embeddings[:, 0], top_embeddings[:, 1],
               c='black', s=150, alpha=0.6, edgecolors='yellow', linewidth=2,
               marker='o', label=f'Top {top_k} Results', zorder=5)

    # Plot query point
    plt.scatter(query_2d[0], query_2d[1],
               c='red', s=400, alpha=0.9, edgecolors='black', linewidth=2,
               marker='*', label='Query', zorder=10)

    # Draw lines from query to top results
    for idx in top_indices:
        plt.plot([query_2d[0], papers_2d[idx, 0]],
                [query_2d[1], papers_2d[idx, 1]],
                'k--', alpha=0.2, linewidth=1, zorder=1)

    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title(f'Search Results for: "{query}"\n' +
             '(Query shown as red star, top results highlighted)',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print the top results
    print(f"\nTop {top_k} results for query: '{query}'\n{separator}")
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank}. [{similarities[idx]:.4f}] {df['title'].iloc[idx][:50]}...")
        print(f"   Category: {df['category'].iloc[idx]}")

# Visualize search results for a query
query = "language model pretraining"
visualize_search_results(query, embeddings, df, top_k=10)
```

[![Language Model Pretraining](https://www.dataquest.io/wp-content/uploads/2025/11/Language-Model-Pretraining.png.webp)](https://www.dataquest.io/wp-content/uploads/2025/11/Language-Model-Pretraining.png.webp)

```
Top 10 results for query: 'language model pretraining'
================================================================================
1. [0.4240] Reusing Pre-Training Data at Test Time is a Comput...
   Category: cs.CL
2. [0.4102] Evo-1: Lightweight Vision-Language-Action Model wi...
   Category: cs.CV
3. [0.3910] PixCLIP: Achieving Fine-grained Visual Language Un...
   Category: cs.CV
4. [0.3713] PLLuM: A Family of Polish Large Language Models...
   Category: cs.CL
5. [0.3712] SCALE: Upscaled Continual Learning of Large Langua...
   Category: cs.CL
6. [0.3528] Q3R: Quadratic Reweighted Rank Regularizer for Eff...
   Category: cs.LG
7. [0.3334] LLMs and Cultural Values: the Impact of Prompt Lan...
   Category: cs.CL
8. [0.3297] TwIST: Rigging the Lottery in Transformers with In...
   Category: cs.LG
9. [0.3278] IndicSuperTokenizer: An Optimized Tokenizer for In...
   Category: cs.CL
10. [0.3157] Bearing Syntactic Fruit with Stack-Augmented Neura...
   Category: cs.CL
```

This visualization reveals exactly what's happening during semantic search. The red star represents your query embedding. The black circles highlighted in yellow are the top 10 results. The dotted lines connect the query to its top 10 matches, showing the spatial relationships.

Notice how most of the top results cluster near the query in embedding space. The majority are from the Computational Linguistics category (the green cluster), which makes perfect sense for a query about language model pretraining. Papers from other categories sit farther away in the visualization, corresponding to their lower similarity scores.

You might notice some papers that appear visually closer to the red query star aren't in our top 10 results. This happens because PCA compresses 1536 dimensions down to just 2 for visualization. This lossy compression can't perfectly preserve all distance relationships from the original high-dimensional space. The similarity scores we display are calculated in the full 1536-dimensional embedding space before PCA, which is why they're more accurate than visual proximity in this 2D plot. Think of the visualization as showing general clustering patterns rather than exact rankings.

This spatial representation makes the abstract concept of similarity concrete. High similarity scores mean points that are close together in the original high-dimensional embedding space. When we say two papers are semantically similar, we're saying their embeddings point in similar directions.

Let's try another visualization with a different query:

```python
# Try a more specific query
query = "reinforcement learning algorithms"
visualize_search_results(query, embeddings, df, top_k=10)
```

[![Reinforcement Learning Algorithms](https://www.dataquest.io/wp-content/uploads/2025/11/Reinforcement-Learning-Algorithms.png.webp)](https://www.dataquest.io/wp-content/uploads/2025/11/Reinforcement-Learning-Algorithms.png.webp)

```
Top 10 results for query: 'reinforcement learning algorithms'
================================================================================
1. [0.3477] Exchange Policy Optimization Algorithm for Semi-In...
   Category: cs.LG
2. [0.3429] Fitting Reinforcement Learning Model to Behavioral...
   Category: cs.LG
3. [0.3091] Online Algorithms for Repeated Optimal Stopping: A...
   Category: cs.LG
4. [0.3062] DeepPAAC: A New Deep Galerkin Method for Principal...
   Category: cs.LG
5. [0.2970] Environment Agnostic Goal-Conditioning, A Study of...
   Category: cs.LG
6. [0.2925] Forgetting is Everywhere...
   Category: cs.LG
7. [0.2865] RLHF: A comprehensive Survey for Cultural, Multimo...
   Category: cs.CL
8. [0.2857] RLHF: A comprehensive Survey for Cultural, Multimo...
   Category: cs.LG
9. [0.2827] End-to-End Reinforcement Learning of Koopman Model...
   Category: cs.LG
10. [0.2813] GrowthHacker: Automated Off-Policy Evaluation Opti...
   Category: cs.SE
```

This visualization shows clear clustering around the Machine Learning region (red cluster), and most top results are ML papers about reinforcement learning. The query star lands right in the middle of where we'd expect for an RL-focused query, and the top results fan out from there in the embedding space.

Use these visualizations to spot broad trends (like whether your query lands in the right category cluster), not to validate exact rankings. The rankings come from measuring distances in all 1536 dimensions, while the visualization shows only 2.

## Evaluating Search Quality

How do we know if our search system is working well? In production systems, you'd use quantitative metrics like Precision@K, Recall@K, or [Mean Average Precision (MAP)](https://towardsdatascience.com/mean-average-precision-at-k-map-k-clearly-explained-538d8e032d2/). These metrics require labeled relevance judgments where humans mark which papers are relevant for specific queries.

For this tutorial, we'll use qualitative evaluation. Let's examine results for a query and assess whether they make sense:

```python
# Detailed evaluation of a single query
query = "anomaly detection techniques"
results = semantic_search(query, embeddings, df, top_k=10)

print(f"Query: '{query}'\n")
print(f"Detailed Results:\n{separator}")

for rank, (idx, row) in enumerate(results.iterrows(), 1):
    print(f"\nRank {rank} | Similarity: {row['similarity_score']:.4f}")
    print(f"Title: {row['title']}")
    print(f"Category: {row['category']}")
    print(f"Abstract: {row['abstract'][:200]}...")
    print("-" * 80)
```
```
Query: 'anomaly detection techniques'

Detailed Results:
================================================================================

Rank 1 | Similarity: 0.3895
Title: An Encode-then-Decompose Approach to Unsupervised Time Series Anomaly Detection on Contaminated Training Data--Extended Version
Category: cs.DB
Abstract: Time series anomaly detection is important in modern large-scale systems and is applied in a variety of domains to analyze and monitor the operation of
diverse systems. Unsupervised approaches have re...
--------------------------------------------------------------------------------

Rank 2 | Similarity: 0.3268
Title: DeNoise: Learning Robust Graph Representations for Unsupervised Graph-Level Anomaly Detection
Category: cs.LG
Abstract: With the rapid growth of graph-structured data in critical domains,
unsupervised graph-level anomaly detection (UGAD) has become a pivotal task.
UGAD seeks to identify entire graphs that deviate from ...
--------------------------------------------------------------------------------

Rank 3 | Similarity: 0.3218
Title: IEC3D-AD: A 3D Dataset of Industrial Equipment Components for Unsupervised Point Cloud Anomaly Detection
Category: cs.CV
Abstract: 3D anomaly detection (3D-AD) plays a critical role in industrial
manufacturing, particularly in ensuring the reliability and safety of core
equipment components. Although existing 3D datasets like Rea...
--------------------------------------------------------------------------------

Rank 4 | Similarity: 0.3085
Title: Conditional Score Learning for Quickest Change Detection in Markov Transition Kernels
Category: cs.LG
Abstract: We address the problem of quickest change detection in Markov processes with unknown transition kernels. The key idea is to learn the conditional score
$nabla_{\mathbf{y}} \log p(\mathbf{y}|\mathbf{x...
--------------------------------------------------------------------------------

Rank 5 | Similarity: 0.3053
Title: Multiscale Astrocyte Network Calcium Dynamics for Biologically Plausible Intelligence in Anomaly Detection
Category: cs.LG
Abstract: Network anomaly detection systems encounter several challenges with
traditional detectors trained offline. They become susceptible to concept drift
and new threats such as zero-day or polymorphic atta...
--------------------------------------------------------------------------------

Rank 6 | Similarity: 0.2907
Title: I Detect What I Don't Know: Incremental Anomaly Learning with Stochastic Weight Averaging-Gaussian for Oracle-Free Medical Imaging
Category: cs.CV
Abstract: Unknown anomaly detection in medical imaging remains a fundamental challenge due to the scarcity of labeled anomalies and the high cost of expert
supervision. We introduce an unsupervised, oracle-free...
--------------------------------------------------------------------------------

Rank 7 | Similarity: 0.2901
Title: Adaptive Detection of Software Aging under Workload Shift
Category: cs.SE
Abstract: Software aging is a phenomenon that affects long-running systems, leading to progressive performance degradation and increasing the risk of failures. To
mitigate this problem, this work proposes an ad...
--------------------------------------------------------------------------------

Rank 8 | Similarity: 0.2763
Title: The Impact of Data Compression in Real-Time and Historical Data Acquisition Systems on the Accuracy of Analytical Solutions
Category: cs.DB
Abstract: In industrial and IoT environments, massive amounts of real-time and
historical process data are continuously generated and archived. With sensors
and devices capturing every operational detail, the v...
--------------------------------------------------------------------------------

Rank 9 | Similarity: 0.2570
Title: A Large Scale Study of AI-based Binary Function Similarity Detection Techniques for Security Researchers and Practitioners
Category: cs.SE
Abstract: Binary Function Similarity Detection (BFSD) is a foundational technique in software security, underpinning a wide range of applications including
vulnerability detection, malware analysis. Recent adva...
--------------------------------------------------------------------------------

Rank 10 | Similarity: 0.2418
Title: Fraud-Proof Revenue Division on Subscription Platforms
Category: cs.LG
Abstract: We study a model of subscription-based platforms where users pay a fixed fee for unlimited access to content, and creators receive a share of the revenue. Existing approaches to detecting fraud predom...
--------------------------------------------------------------------------------
```

Looking at these results, we can assess quality by asking:

- **Are the results relevant to the query?** Yes! All papers discuss anomaly detection techniques and methods.
- **Are similarity scores meaningful?** Yes! Higher-ranked papers are more directly relevant to the query.
- **Does the ranking make sense?** Yes! The top result is specifically about time series anomaly detection, which directly matches our query.

Let's look at what similarity score thresholds might indicate:

```python
# Analyze the distribution of similarity scores
query = "query optimization algorithms"
results = semantic_search(query, embeddings, df, top_k=50)

print(f"Query: '{query}'")
print(f"\nSimilarity score distribution for top 50 results:")
print(f"  Highest score: {results['similarity_score'].max():.4f}")
print(f"  Median score: {results['similarity_score'].median():.4f}")
print(f"  Lowest score: {results['similarity_score'].min():.4f}")

# Show how scores change with rank
print(f"\nScore decay by rank:")
for rank in [1, 5, 10, 20, 30, 40, 50]:
    score = results['similarity_score'].iloc[rank-1]
    print(f"  Rank {rank:2d}: {score:.4f}")
```
```
Query: 'query optimization algorithms'

Similarity score distribution for top 50 results:
  Highest score: 0.4206
  Median score: 0.2765
  Lowest score: 0.2402

Score decay by rank:
  Rank  1: 0.4206
  Rank  5: 0.3666
  Rank 10: 0.3144
  Rank 20: 0.2910
  Rank 30: 0.2737
  Rank 40: 0.2598
  Rank 50: 0.2402
```

Similarity score interpretation depends heavily on your dataset characteristics. Here are general heuristics, but they require adjustment based on your specific data:

**For broad, multi-domain datasets (like ours with 5 distinct categories):**

- 0.40+: Highly relevant
- 0.30-0.40: Very relevant
- 0.25-0.30: Moderately relevant
- Below 0.25: Questionable relevance

**For narrow, specialized datasets (single domain):**

- 0.70+: Highly relevant
- 0.60-0.70: Very relevant
- 0.50-0.60: Moderately relevant
- Below 0.50: Questionable relevance

The key is understanding relative rankings within your dataset rather than treating these as universal thresholds. Our multi-domain dataset naturally produces lower absolute scores than a specialized single-topic dataset would. What matters is that the top results are genuinely more relevant than lower-ranked results.

## Testing Edge Cases

A good search system should handle different types of queries gracefully. Let's test some edge cases:

```python
# Test 1: Very specific technical query
print(f"Test 1: Highly Specific Query\n{separator}")
query = "graph neural networks for molecular property prediction"
results = semantic_search(query, embeddings, df, top_k=3)
print(f"Query: '{query}'\n")
for idx, row in results.iterrows():
    print(f"  [{row['similarity_score']:.4f}] {row['title'][:50]}...")

# Test 2: Broad general query
print(f"\n\nTest 2: Broad General Query\n{separator}")
query = "artificial intelligence"
results = semantic_search(query, embeddings, df, top_k=3)
print(f"Query: '{query}'\n")
for idx, row in results.iterrows():
    print(f"  [{row['similarity_score']:.4f}] {row['title'][:50]}...")

# Test 3: Query with common words
print(f"\n\nTest 3: Common Words Query\n{separator}")
query = "learning from data"
results = semantic_search(query, embeddings, df, top_k=3)
print(f"Query: '{query}'\n")
for idx, row in results.iterrows():
    print(f"  [{row['similarity_score']:.4f}] {row['title'][:50]}...")
```
```
Test 1: Highly Specific Query
================================================================================
Query: 'graph neural networks for molecular property prediction'

  [0.3602] ScaleDL: Towards Scalable and Efficient Runtime Pr...
  [0.3072] RELATE: A Schema-Agnostic Perceiver Encoder for Mu...
  [0.3032] Dark Energy Survey Year 3 results: Simulation-base...

Test 2: Broad General Query
================================================================================
Query: 'artificial intelligence'

  [0.3202] Lessons Learned from the Use of Generative AI in E...
  [0.3137] AI for Distributed Systems Design: Scalable Cloud ...
  [0.3096] SmartMLOps Studio: Design of an LLM-Integrated IDE...

Test 3: Common Words Query
================================================================================
Query: 'learning from data'

  [0.2912] PrivacyCD: Hierarchical Unlearning for Protecting ...
  [0.2879] Learned Static Function Data Structures...
  [0.2732] REMIND: Input Loss Landscapes Reveal Residual Memo...
```

Notice what happens:

- **Specific queries** return focused, relevant results with higher similarity scores
- **Broad queries** return more general papers about AI with moderate scores
- **Common word queries** still find relevant content because embeddings understand context

This demonstrates the power of semantic search over keyword matching. A keyword search for "learning from data" would match almost everything, but semantic search understands the intent and returns papers about data-driven learning and optimization.

## Understanding Retrieval Quality

Let's create a function to help us understand why certain papers are retrieved for a query:

```python
def explain_search_result(query, paper_idx, embeddings, df):
    """
    Explain why a particular paper was retrieved for a query.
    """
    # Get query embedding
    response = co.embed(
        texts=[query],
        model='embed-v4.0',
        input_type='search_query',
        embedding_types=['float']
    )
    query_embedding = np.array(response.embeddings.float_[0])

    # Calculate similarity
    paper_embedding = embeddings[paper_idx]
    similarity = cosine_similarity(
        query_embedding.reshape(1, -1),
        paper_embedding.reshape(1, -1)
    )[0][0]

    # Show the result
    print(f"Query: '{query}'")
    print(f"\nPaper: {df['title'].iloc[paper_idx]}")
    print(f"Category: {df['category'].iloc[paper_idx]}")
    print(f"Similarity Score: {similarity:.4f}")
    print(f"\nAbstract:")
    print(df['abstract'].iloc[paper_idx][:300] + "...")

    # Show how this compares to all papers
    all_similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]
    rank = (all_similarities > similarity).sum() + 1

    print(f"\nRanking: {rank}/{len(df)} papers")
    percentage = (len(df) - rank)/len(df)*100
    print(f"This paper is more relevant than {percentage:.1f}% of papers")

# Explain why a specific paper was retrieved
query = "database query optimization"
paper_idx = 322
explain_search_result(query, paper_idx, embeddings, df)
```
```
Query: 'database query optimization'

Paper: L2T-Tune:LLM-Guided Hybrid Database Tuning with LHS and TD3
Category: cs.DB
Similarity Score: 0.3374

Abstract:
Configuration tuning is critical for database performance. Although recent
advancements in database tuning have shown promising results in throughput and
latency improvement, challenges remain. First, the vast knob space makes direct
optimization unstable and slow to converge. Second, reinforcement ...

Ranking: 9/500 papers
This paper is more relevant than 98.2% of papers
```

This explanation shows exactly why the paper was retrieved: it has a solid similarity score (0.3373) and ranks in the top 2% of all papers for this query. The abstract clearly discusses database configuration tuning and optimization, which matches the query intent perfectly.

## Applying These Skills to Your Own Projects

You now have a complete semantic search system. The skills you've learned here transfer directly to any domain where you need to find relevant documents based on meaning.

**The pattern is always the same:**

1. Collect documents (APIs, databases, file systems)
2. Generate embeddings (local models or API services)
3. Store embeddings efficiently (files or vector databases)
4. Implement similarity calculations (cosine, dot product, or Euclidean)
5. Build a search function that returns ranked results
6. Evaluate results to ensure quality

This exact workflow applies whether you're building:

- A research paper search engine (what we just built)
- A code search system for documentation
- A customer support knowledge base
- A product recommendation system
- A legal document retrieval system

The only difference is the data source. Everything else remains the same.

## Optimizing for Production

Before we wrap up, let's discuss a few optimizations for production systems. We won't implement these now, but knowing they exist will help you scale your search system when needed.

### 1\. Caching Query Embeddings

If users frequently search for similar queries, caching the embeddings can save significant API calls and computation time. Store the query text and its embedding in memory or a database. When a user searches, check if you've already generated an embedding for that exact query. This simple optimization can reduce costs and improve response times, especially for popular searches.

### 2\. Approximate Nearest Neighbors

For datasets with millions of embeddings, exact similarity calculations become slow. Libraries like FAISS, Annoy, or ScaNN provide approximate nearest neighbor search that's much faster. These specialized libraries use clever indexing techniques to quickly find embeddings that are close to your query without calculating the exact distance to every single vector in your database. While we didn't implement this in our tutorial series, it's worth knowing that these tools exist for production systems handling large-scale search.

### 3\. Batch Query Processing

When processing multiple queries, batch them together for efficiency. Instead of generating embeddings one query at a time, send multiple queries to your embedding API in a single request. Most embedding APIs support batch processing, which reduces network overhead and can be significantly faster than sequential processing. This approach is particularly valuable when you need to process many queries at once, such as during system evaluation or when handling multiple concurrent users.

### 4\. Vector Database Storage

For production systems, use vector databases (Pinecone, Weaviate, Chroma) rather than files. Vector databases handle indexing, similarity search optimization, and storage efficiency automatically. They also apply float32 precision by default for memory efficiency. Something you'd need to handle manually with file-based storage (converting from float64 to float32 can halve storage requirements with minimal impact on search quality).

### 5\. Document Chunking for Long Content

In our tutorial, we embedded entire paper abstracts as single units. This works fine for abstracts (which are naturally concise), but production systems often process longer documents like full papers, documentation, or articles. Industry best practice is to chunk these into coherent sections (typically 200-1000 tokens per chunk) for optimal semantic fidelity. This ensures each embedding captures a focused concept rather than trying to represent an entire document's diverse topics in a single vector. Modern models with high token limits (8k+ tokens) make this less critical than before, but chunking still improves retrieval quality for longer content.

These optimizations become critical as your system scales, but the core concepts remain the same. Start with the straightforward implementation we've built, then add these optimizations when performance or cost becomes a concern.

## What You've Accomplished

You've built a complete semantic search system from the ground up! Let's review what you've learned:

1. You understand three distance metrics (Euclidean distance, dot product, cosine similarity) and when to use each one.
2. You can implement similarity calculations both manually (to understand the math) and efficiently (using scikit-learn).
3. You've built a search function that converts queries to embeddings and returns ranked results.
4. You can visualize search results in embedding space to understand spatial relationships between queries and documents.
5. You can evaluate search quality qualitatively by examining results and similarity scores.
6. You understand how to optimize search systems for production with caching, approximate search, batching, vector databases, and document chunking.

Most importantly, you now have the complete skillset to build your own search engine. You know how to:

- **Collect data from APIs**
- **Generate embeddings**
- **Calculate semantic similarity**
- **Build and evaluate search systems**

## Next Steps

Before moving on, try experimenting with your search system:

**Test different query styles:**

- Try very short queries ("neural nets") vs detailed queries ("applying deep neural networks to computer vision tasks")
- See how the system handles questions vs keywords
- Test queries that combine multiple topics

**Explore the similarity threshold:**

- Set a minimum similarity threshold (e.g., 0.30) and see how many results pass
- Test what happens with a very strict threshold (0.40+)
- Find the sweet spot for your use case

**Analyze failure cases:**

- Find queries where the results aren't great
- Understand why (too broad? too specific? wrong domain?)
- Think about how you'd improve the system

**Compare categories:**

- Search for "deep learning" and see which categories dominate results
- Try category-specific searches and verify papers match
- Look for interesting cross-category papers

**Visualize different queries:**

- Create visualizations for queries from different domains
- Observe how the query point moves in embedding space
- Notice which categories cluster near different types of queries

This experimentation will sharpen your intuition about how semantic search works and prepare you to debug issues in your own projects.

---

**Key Takeaways:**

- Euclidean distance measures straight-line distance between vectors and is the most intuitive metric
- Dot product multiplies corresponding elements and is computationally efficient
- Cosine similarity measures the angle between vectors and is the standard for text embeddings
- For well-normalized embeddings, all three metrics typically produce similar rankings
- Similarity scores depend on dataset characteristics and should be interpreted relative to your specific data
- Multi-domain datasets naturally produce lower absolute scores than specialized single-topic datasets
- Visualizing search results in 2D embedding space helps understand clustering patterns, though exact rankings come from the full high-dimensional space
- The spatial proximity of embeddings directly corresponds to semantic similarity scores
- Production search systems benefit from query caching, approximate nearest neighbors, batch processing, vector databases, and document chunking
- The semantic search pattern (collect, embed, calculate similarity, rank) applies universally across domains
- Qualitative evaluation through manual inspection is crucial for understanding search quality
- Edge cases like very broad or very specific queries test the robustness of your search system
- These skills transfer directly to building search systems in any domain with any type of content