from src.summarization.llm_summarization import summarize_all_topics
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os 

def send_email():
    load_dotenv()
    #topic_summaries = summarize_all_topics()

    sender_email = os.getenv("EMAIL_ADDRESS")
    receiver_email = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("APP_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = "Weekly Digest of PubMed"
    msg["From"] = sender_email
    msg["To"] = receiver_email 

    msg.set_content("Please use a browser that supports HTML content. Thank you!")
    html_content = """
        <!DOCTYPE html>
        <html>
            <body>
                <h1 style="color: #4CAF50;">Hello World</h1>
            </body>
        </html>
    """
    msg.add_alternative(html_content, subtype="html")

    try: 
        with smtplib.SMTP("smtp.gmail.com", 587) as server: 
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e: 
        print(f"Failed to send email: {e}")

send_email()