from Bio import Entrez
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

Entrez.email = os.getenv("EMAIL_ADDRESS")

# Date from a month ago 
last_week = datetime.now() - timedelta(days=30)
date_last_week = last_week.strftime('%Y/%m/%d')

# Downloading PubMed papers
query = date_last_week + "[Date - Publication] : 3000[Date - Publication]"
stream = Entrez.esearch(db = "pubmed", term = query, retmax = 2000, sort = "relevance")
record = Entrez.read(stream)
stream.close()
idlist = record["IdList"]
stream = Entrez.efetch(db="pubmed", id=idlist, rettype="abstract", retmode="xml")
records = Entrez.read(stream)
stream.close()

# Parsing 
papers = []

for record in records["PubmedArticle"]:
    try: 
        medline = record["MedlineCitation"]
        article = medline["Article"]
        title = str(article["ArticleTitle"])[:-1]

        # Only keep valid publication types 
        valid_pub_types = {
            "Journal Article",
            "Preprint",
            "Review",
            "Comparative Study",
            "Systematic Review",
            "Observational Study",
            "Validation Study",
            "Multicenter Study",
            "Meta-Analysis",
            "Randomized Controlled Trial",
            "Case Reports"
        }

        pub_types = []
        if "PublicationTypeList" in article:
            for pt in article["PublicationTypeList"]:
                pub_types.append(str(pt))
        
        if not any(pt in valid_pub_types for pt in pub_types):
            continue

        abstract = ""
        if "Abstract" not in article: 
            continue

        abstract_parts = []
        for text in article["Abstract"]["AbstractText"]:
            abstract_parts.append(str(text))
        abstract = " ".join(abstract_parts)

        journal = str(article["Journal"]["Title"])
        pub_date = article["Journal"]["JournalIssue"]["PubDate"]
        mesh_tags = []
        if "MeshHeadingList" in medline:
            for mesh in medline["MeshHeadingList"]:
                mesh_tags.append(str(mesh["DescriptorName"]))
        
        papers.append({
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "publication_date": pub_date,
            "mesh_tags": mesh_tags
        })
    except KeyError:
        continue