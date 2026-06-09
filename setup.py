from dotenv import load_dotenv
import os

load_dotenv()

required = ["GEMINI_API_KEY", "EMAIL_ADDRESS", "APP_PASSWORD"]
missing = [key for key in required if not os.getenv(key)]

if missing:
    print(f"Missing required environment variables: {missing}")
    print("Please check your .env file")
else:
    print("Setup complete — all environment variables found")