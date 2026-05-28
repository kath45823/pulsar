from transformers import pipeline
import re

diseases_ner = pipeline("ner", model="OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M", aggregation_strategy = "simple")
cell_ner = pipeline("ner", model="siddharthtumre/biobert-finetuned-ner", aggregation_strategy = "simple")
chem_ner = pipeline("ner", model="OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M", aggregation_strategy = "first")
gene_ner = pipeline("ner", model="pruas/BENT-PubMedBERT-NER-Gene", aggregation_strategy = "first")

test_abstract = """
Alzheimer's disease (AD) is characterized by progressive neurodegeneration driven by 
amyloid-beta plaques and tau protein hyperphosphorylation. We investigated the 
therapeutic potential of donepezil and memantine combination therapy in APP/PS1 
transgenic mice. RNA sequencing of hippocampal neurons revealed significant 
upregulation of APOE and TREM2 gene expression, while BACE1 mRNA levels were 
markedly reduced following treatment. Immunohistochemical analysis of HT22 and 
SH-SY5Y cell lines demonstrated decreased amyloid precursor protein (APP) levels 
in microglia and astrocytes. Our findings suggest that combination therapy 
significantly reduces tau phosphorylation and neuroinflammation in patients with 
mild cognitive impairment (MCI).
"""

def clean(word):
    word = re.sub(r'\s*-\s*', '-', word)
    word = re.sub(r'\s*cell\s*(line|type|lines|types).*', '', word, flags=re.IGNORECASE)
    word = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', word)
    word = re.sub(r'\s+', ' ', word)
    word = re.sub(r'\s+(protein|gene|mrna|rna)$', '', word, flags=re.IGNORECASE)

    return word.strip()

def abstract_ner(abstract):
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


print(abstract_ner(test_abstract))