# app/main.py
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
from ddgs import DDGS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ============================
# Setup
# ============================
BASE_DIR = os.path.dirname(__file__)
env_path = os.path.join(BASE_DIR, "..", ".env")
load_dotenv(dotenv_path=env_path)

KEY = os.environ.get("OPENAI_API_KEY")
if not KEY:
    raise RuntimeError(f"❌ OPENAI_API_KEY not found. Make sure it's set in {env_path}")

client = OpenAI(api_key=KEY)

app = FastAPI(title="Research Assistant")

# ============================
# Utility: Web Search
# ============================
def web_search(query: str, max_results: int = 5) -> list:
    results = list(DDGS().text(query, max_results=max_results))
    return [
        {
            "title": r.get("title"),
            "url": r.get("href"),
            "snippet": r.get("body"),
        }
        for r in results
    ]

def needs_web_search(prompt: str) -> bool:
    """Ask the model if this query requires external / real-time info."""
    decision_prompt = f"""
    Decide if this query requires external or recent information 
    (like news, current events, or recent studies). 
    Answer only 'Yes' or 'No'.

    Query: {prompt}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a classifier."},
            {"role": "user", "content": decision_prompt},
        ],
        max_tokens=1,
    )
    answer = resp.choices[0].message.content.strip().lower()
    return "yes" in answer

# ============================
# Routes
# ============================

@app.get("/")
async def home():
    """Serve the main HTML UI."""
    file_path = os.path.join(BASE_DIR, "new-layout.html")
    with open(file_path) as f:
        return HTMLResponse(f.read())

@app.get("/references")
async def references_page():
    """Serve references page."""
    file_path = os.path.join(BASE_DIR, "references.html")
    with open(file_path) as f:
        return HTMLResponse(f.read())


# Setup Pinecone + embeddings
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_KEY:
    raise RuntimeError("❌ PINECONE_API_KEY not found in .env")

pc = Pinecone(api_key=PINECONE_KEY)
embed_model = SentenceTransformer(
    model_name_or_path="nomic-ai/nomic-embed-text-v2-moe",
    trust_remote_code=True
)

# ============================
# Q&A Endpoint
# ============================
@app.post("/api/ask")
async def ask(req: Request):
    """Main Q&A endpoint."""
    data = await req.json()
    prompt = data.get("prompt", "")
    use_web = data.get("use_web_search", None)  # true/false/null
    num_results = data.get("num_search_results", 3)
    transcript = data.get("transcript", "").strip()

    if not prompt:
        return JSONResponse({"error": "Prompt is required."}, status_code=400)

    # Decide if web search is required
    try:
        if use_web is None:  
            use_web = needs_web_search(prompt)
        elif use_web is True:
            use_web = True
        else:
            use_web = False
    except Exception as e:
        return JSONResponse({"error": f"Web search classifier failed: {e}"}, status_code=500)

    # If web search requested
    context = ""
    if use_web:
        try:
            results = web_search(prompt, num_results)
            if results:
                context = "Web search results:\n"
                for i, r in enumerate(results, 1):
                    context += f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}\n\n"
                prompt = (
                    f"{context}\n\n"
                    "Using the web search results above, answer the user’s query. "
                    "Always cite relevant results with their number and include URLs."
                    f"\n\nUser query: {prompt}"
                )
            else:
                return JSONResponse({"reply": "No web results found.", "used_web": True})
        except Exception as e:
            return JSONResponse({"error": f"Web search failed: {e}"}, status_code=500)

    # If transcript provided
    if transcript:
        prompt = (
            f"Here is the transcript of the meeting:\n\n{transcript}\n\n"
            f"Now answer the user’s question based on this transcript: {prompt}"
        )

    sys_prompt = (
        "You are a helpful research assistant. "
        "If web search results are provided, you MUST use them directly. "
        "Always include the links (URLs) in your answer. "
        "If transcript context is provided, incorporate it into your reasoning. "
    )

    messages = [{"role": "system", "content": sys_prompt}]
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages
        )
        reply = response.choices[0].message.content
        return JSONResponse({"reply": reply, "used_web": use_web})
    except Exception as e:
        return JSONResponse({"error": f"OpenAI API failed: {e}"}, status_code=500)