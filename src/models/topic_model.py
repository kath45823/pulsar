from src.features.scores import filtered_papers
from bertopic.representation import ZeroShotClassification
from bertopic import BERTopic

papers = filtered_papers()
mesh_tags = []
abstracts = []

for paper in papers:
    mesh_tags.append(paper["mesh_tags"])
    abstracts.append(paper["abstract"])

mesh_tags = [t for tags in mesh_tags for t in tags]

# representation_model = ZeroShotClassification(mesh_tags, model="facebook/bart-large-mnli")
# topic_model = BERTopic(representation_model = representation_model)
topic_model = BERTopic()
topics, prob = topic_model.fit_transform(abstracts)
print(topic_model.get_topic_info())