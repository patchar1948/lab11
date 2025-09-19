import os
from dotenv import load_dotenv
import litellm
load_dotenv()


MODEL = os.getenv("MODEL")
if not MODEL:
   raise SystemExit("Please set MODEL in your .env (e.g., groq/llama-3.3-70b-versatile).")	




EMBED_MODEL = os.getenv("EMBED_MODEL")
if not EMBED_MODEL:
   raise SystemExit("Please set EMBED_MODEL in your .env (e.g., sentence-transformers/all-MiniLM-L6-v2).")
