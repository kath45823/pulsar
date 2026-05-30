from src.features.scores import filtered_papers
from bertopic.representation import ZeroShotClassification
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import random 
import numpy as np

def get_papers_per_topic():
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
        "Ophthalmology",
        "Psychiatry",
        "Drug Resistance",
        "Vaccines",
        "Epidemiology",
        "Traditional Medicine",
        "Structural Biology",

        # biology
        "Genomics",
        "Microbiology",
        "Plant Biology",
        "Ecology",
        "Biochemistry",
        "Aging",
        "Food Science",

        # chemistry
        "Organic Chemistry",
        "Materials Science",
        "Catalysis",
        "Electrochemistry",
        "Spectroscopy",

        # environmental
        "Climate Change",
        "Environmental Pollution",

        # physics & engineering
        "Quantum Mechanics",
        "Astrophysics",
        "Biomaterials",
        "Energy Storage",
        "Superconductivity",

        # computational
        "Machine Learning",
        "Bioinformatics",
        "Computer Science",
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
    topic_papers = {}

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        title = row["Name"]

        if tid == -1:
            continue

        papers_per_topic = []
        for paper, topic_id in zip(papers, topics):
            if topic_id == tid:
                papers_per_topic.append(paper)

        topic_papers[title] = papers_per_topic

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    topic_model.save("src/models/topic_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

    return topic_papers