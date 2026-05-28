from transformers import pipeline

diseases_ner = pipeline("ner", model="OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M", aggregation_strategy = "simple")
cell_ner = pipeline("ner", model="siddharthtumre/biobert-finetuned-ner", aggregation_strategy = "simple")
chem_ner = pipeline("ner", model="OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M", aggregation_strategy = "first")
gene_ner = pipeline("ner", model="pruas/BENT-PubMedBERT-NER-Gene", aggregation_strategy = "first")

test_abstract = """
Checkpoint inhibitor therapy with pembrolizumab has demonstrated significant efficacy 
in patients with advanced non-small cell lung cancer (NSCLC) harboring PD-L1 
overexpression. In this study, we investigated the role of KRAS and TP53 mutations 
in modulating immune response in HeLa and MCF-7 cell lines. Treatment with 
cisplatin and pembrolizumab combination showed synergistic tumor suppression 
compared to monotherapy. Western blot analysis revealed upregulation of PD-L1 
protein and downregulation of tumor necrosis factor mRNA (TNF mRNA) in CD8+ T cells. 
Our results demonstrate that KRAS mutation significantly reduces immunotherapy 
efficacy in NSCLC patients with concurrent BRCA1 alterations.
"""

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
            entities["diseases"].append(entity["word"])
    
    for entity in cell_ner(abstract):
        if entity["score"] >= 0.9:
            if entity["entity_group"] == "cell_type":
                entities["cell_types"].append(entity["word"])
            elif entity["entity_group"] == "cell_line":
                entities["cell_lines"].append(entity["word"])
    
    for entity in chem_ner(abstract):
        if entity["score"] >= 0.9 and entity["entity_group"] == "CHEM":
            entities["chemicals"].append(entity["word"])

    for entity in gene_ner(abstract):
        if entity["score"] >= 0.9 and entity["entity_group"] == "B":
            entities["genes/proteins"].append(entity["word"])

    for e_type in entities:
        entities[e_type] = list(set(entities[e_type]))
    
    return entities


print(abstract_ner(test_abstract))