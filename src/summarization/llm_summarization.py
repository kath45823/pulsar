from src.models.ner_model import run_ner_on_topics
from google import genai
from dotenv import load_dotenv
import os

def clean_title(title, rep_abstracts, client):
    parts = title.split("_")
    is_keyword = False
    if "___" not in title and len(parts) > 2:
        is_keyword = True
    
    if is_keyword: 
        prompt = f"""
            Based on these keywords and research abstracts, generate a short 2-4 word scientific domain title.

            Keywords: {title}
            Abstracts: {rep_abstracts[:500]}

            Respond with only the title, nothing else.
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text.strip()
    else: 
        return title.split("___")[0].split("_", 1)[1]

def summarize_all_topics():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    topic_papers_tags = run_ner_on_topics()
    topic_summaries = {}

    for topic, topic_item in topic_papers_tags.items():
        rep_abstracts = "\n\n".join(topic_item["representative_abstracts"])
        title = clean_title(topic, rep_abstracts, client)
        count = topic_item["paper_count"]
        chem = topic_item["tags"]["chemicals"]
        diseases = topic_item["tags"]["diseases"]
        genes = topic_item["tags"]["genes/proteins"]
        cell_types = topic_item["tags"]["cell_types"]
        cell_lines = topic_item["tags"]["cell_lines"]

        prompt = f"""
            Topic: {title}
            Papers this week: {count}

            Key entities:
            - Chemicals: {chem}
            - Diseases: {diseases}  
            - Genes/Proteins: {genes}
            - Cell types: {cell_types}
            - Cell lines: {cell_lines}

            Representative abstracts:
            {rep_abstracts}

            Write a 3-4 sentence summary of what research is saying about 
            this topic this week. Be specific, reference key entities
        """
        
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )

        topic_summaries[title] = {
            "tags": topic_item["tags"],
            "summary": response.text
        }

    return topic_summaries

print(summarize_all_topics())