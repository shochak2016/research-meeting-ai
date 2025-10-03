import os
from dotenv import load_dotenv
from openai import OpenAI
from ddgs import DDGS  # DuckDuckGo search client

load_dotenv()

KEY = os.environ.get("OPENAI_API_KEY")

"""
LLM class — uses OpenAI for reasoning and DDGS for web search.
This version does NOT require mcp_client.
"""
class LLM:
    def __init__(self, openai_api_key, system=None):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = "gpt-4o-mini"
        self.system = system

    def web_search(self, query: str, max_results: int = 5) -> list:
        """Simple DuckDuckGo search using ddgs package."""
        results = list(DDGS().text(query, max_results=max_results))
        return [
            {
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
            }
            for r in results
        ]

    def ask(self, prompt, use_web_search=False, num_search_results=3, **kwargs):
        """Ask the LLM, optionally with live web search context."""
        context = ""
        if use_web_search:
            results = self.web_search(prompt, num_search_results)
            context = "Web search results:\n"
            for i, r in enumerate(results, 1):
                context += f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}\n\n"
            prompt = f"{context}\n\nBased on these results, {prompt}"

        messages = [{"role": "system", "content": self.system}] if self.system else []
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content