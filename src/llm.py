import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY not found in environment variables!")

llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-large-latest",
    temperature=0.1,
    top_p=0.8,
    streaming=True,
)

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY
)