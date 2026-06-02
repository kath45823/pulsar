from dotenv import load_dotenv
from src.features.preprocessor import fetch_papers
from src.features.scores import filtered_papers
from src.models.topic_model import get_papers_per_topic
from src.models.ner_model import run_ner_on_topics
from src.summarization.llm_summarization import summarize_all_topics
from src.delivery.email_delivery import send_email

def main():
    load_dotenv()
    
    topic_papers = get_papers_per_topic()
    print(f"Found {len(topic_papers)} topics")
    topic_ner_tags = run_ner_on_topics(topic_papers)
    topic_summaries = summarize_all_topics(topic_ner_tags)
    send_email(topic_summaries)

    print("Done!")

if __name__ == "__main__":
    main()