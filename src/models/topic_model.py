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
abstracts = []

for paper in papers:
    abstracts.append(paper["abstract"])

specific_tags = [
    # biomedical
    "Neoplasms",
    "Cardiovascular Diseases",
    "Cardiology",
    "Nervous System Diseases",
    "Neurology",
    "Infectious Disease",
    "Metabolic Diseases",
    "Pulmonary Disease",
    "Inflammatory Disease",
    "Genetic Disorders",
    "Immunology",
    "Pharmacology",
    "Neuroscience",
    "Molecular Biology",
    "Cell Biology",
    "Oncology",
    "Ophthalmology",
    "Psychiatry",
    "Drug Resistance",
    "Vaccines",
    "Surgery",
    "Epidemiology",
    "Preventive Medicine",
    "Traditional Medicine",
    "Analytical Chemistry",
    "Ophthalmology",
    "Psychiatry",
    "Traditional Medicine",

    # biology
    "Genomics",
    "Taxonomy",
    "Microbiology",
    "Plant Biology",
    "Ecology",
    "Evolutionary Biology",
    "Biochemistry",
    "Aging",
    "Food Science",
    "Muscle Biology",
    "Structural Biology",
    "Plant Reproductive Biology",

    # chemistry & materials
    "Organic Chemistry",
    "Materials Science",
    "Nanotechnology",
    "Spectroscopy",
    "Catalysis",
    "Construction Materials",

    # environmental
    "Climate Change",
    "Environmental Pollution",
    "Marine Biology",
    "Soil Science",

    # physics & engineering
    "Quantum Mechanics",
    "Astrophysics",
    "Biomaterials",
    "Biomedical Engineering",
    "Energy Storage",
    "Superconductivity",
    "Control Systems"

    # computational
    "Machine Learning",
    "Bioinformatics",
    "Computer Vision",
    "Statistics",
    "Computer Science"
]

representation_model = ZeroShotClassification(
    specific_tags, 
    model = "facebook/bart-large-mnli",
    min_prob = 0.1
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
topic_info = topic_model.get_topic_info()

for _, row in topic_info.iterrows():
    if row["Topic"] == -1:
        continue
    print(f"Topic {row['Topic']}: {row['Name']} | Count: {row['Count']}")
    print(f"  Sample: {row['Representative_Docs'][0][:100]}")
    print()