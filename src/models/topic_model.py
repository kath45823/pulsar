from src.features.scores import filtered_papers
from bertopic.representation import ZeroShotClassification
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import random 
import numpy as np

random.seed(42)
np.random.seed(42)

papers = filtered_papers()
mesh_tags = []
abstracts = []

for paper in papers:
    mesh_tags.append(paper["mesh_tags"])
    abstracts.append(paper["abstract"])

mesh_tags = [t for tags in mesh_tags for t in tags]
best_mesh_tags = []
c = Counter(mesh_tags)

for tag, count in c.most_common(40):
    best_mesh_tags.append(tag)

representation_model = ZeroShotClassification(
    best_mesh_tags, 
    model = "facebook/bart-large-mnli",
    min_prob = 0.3
)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42  
)

hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="eom"
)

vectorizer = CountVectorizer(stop_words="english")

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    representation_model=representation_model,
    calculate_probabilities=False
)
topics, prob = topic_model.fit_transform(abstracts)

print(topic_model.get_topic_info())
print(topic_model.get_topic_info()["Name"].tolist())