from transformers import pipeline
import re
from src.models.topic_model import get_papers_per_topic
from collections import Counter

diseases_ner = pipeline("ner", model="OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M", aggregation_strategy = "simple")
cell_ner = pipeline("ner", model="siddharthtumre/biobert-finetuned-ner", aggregation_strategy = "simple")
chem_ner = pipeline("ner", model="OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M", aggregation_strategy = "first")
gene_ner = pipeline("ner", model="pruas/BENT-PubMedBERT-NER-Gene", aggregation_strategy = "first")

def clean(word):
    word = re.sub(r'\s*-\s*', '-', word)
    word = re.sub(r'\s*cell\s*(line|type|lines|types).*', '', word, flags=re.IGNORECASE)
    word = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', word)
    word = re.sub(r'\s+', ' ', word)
    word = re.sub(r'\s+(protein|gene|mrna|rna)$', '', word, flags=re.IGNORECASE)

    return word.strip()

def run_ner_on_abstract(abstract):
    entities = {
        "chemicals": [],
        "diseases": [],
        "genes/proteins": [],
        "cell_types": [],
        "cell_lines": []
    }

    for entity in diseases_ner(abstract):
        if entity["score"] >= 0.9 and entity["entity_group"] == "DISEASE":
            entities["diseases"].append(clean(entity["word"]))
    
    for entity in cell_ner(abstract):
        if entity["score"] >= 0.9:
            if entity["entity_group"] == "cell_type":
                entities["cell_types"].append(clean(entity["word"]))
            elif entity["entity_group"] == "cell_line":
                entities["cell_lines"].append(clean(entity["word"]))
    
    for entity in chem_ner(abstract):
        if entity["score"] >= 0.9 and entity["entity_group"] == "CHEM":
            entities["chemicals"].append(clean(entity["word"]))

    for entity in gene_ner(abstract):
        if entity["score"] >= 0.9 and entity["entity_group"] == "B":
            entities["genes/proteins"].append(clean(entity["word"]))

    for e_type in entities:
        entities[e_type] = list(set(entities[e_type]))
    
    return entities

def topic_ner():
    topic_papers = get_papers_per_topic()
    topic_ner_tags = {}

    for topic, papers in topic_papers.items():
        all_ner_tags = {
            "chemicals": [],
            "diseases": [],
            "genes/proteins": [],
            "cell_types": [],
            "cell_lines": []
        }
        for paper in papers: 
            paper_tags = run_ner_on_abstract(paper["abstract"])
            all_ner_tags["chemicals"].extend(paper_tags["chemicals"])
            all_ner_tags["diseases"].extend(paper_tags["diseases"])
            all_ner_tags["genes/proteins"].extend(paper_tags["genes/proteins"])
            all_ner_tags["cell_types"].extend(paper_tags["cell_types"])
            all_ner_tags["cell_lines"].extend(paper_tags["cell_lines"])
        
        chem_counter = Counter(all_ner_tags["chemicals"])
        diseases_counter = Counter(all_ner_tags["diseases"])
        genes_counter = Counter(all_ner_tags["genes/proteins"])
        types_counter = Counter(all_ner_tags["cell_types"])
        lines_counter = Counter(all_ner_tags["cell_lines"])

        topic_tags = {
            "chemicals": [tag for tag, count in chem_counter.most_common(5)],
            "diseases": [tag for tag, count in diseases_counter.most_common(5)],
            "genes/proteins": [tag for tag, count in genes_counter.most_common(5)],
            "cell_types": [tag for tag, count in types_counter.most_common(5)],
            "cell_lines": [tag for tag, count in lines_counter.most_common(5)]
        }

        topic_ner_tags[topic] = topic_tags

    return topic_ner_tags

print(topic_ner())