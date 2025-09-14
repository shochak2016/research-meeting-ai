import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

KEY = os.environ.get("OPENAI_API_KEY")

'''class designed so that OpenAI object created once in streamlit doc then cached so it does nto have to keep reinstatiating'''
class LLM:
    def __init__(self, openai_api_key, mcp_client, system=None):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = "gpt-4o-mini"
        self.system = system
        self.mcp_client = mcp_client

    def fetch_page(self, url: str) -> str:
        """
        Fetch and extract content from a web page using MCP server.
        """
        try:
            content = self.mcp_client.call_tool("fetch_page", {"url": url})
            return content
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"
    
    def web_search(self, query: str, max_results: int = 5, fetch_content: bool = False) -> list:
        """
        Search web using DuckDuckGo MCP server.
        Fetches extra results as buffer in case some pages are inaccessible.
        Returns top max_results (default 5).
        """
        # Fetch 2x the requested results as a buffer, 
        # buffer is used in case LLM is unable to access some websites
        buffer_multiplier = 2
        search_limit = max_results * buffer_multiplier
        
        # Get search results from MCP server
        results = self.mcp_client.call_tool("search", {"query": query, "limit": search_limit})
        
        # Optionally fetch page content for each result
        if fetch_content and results:
            successful_results = []
            for result in results:
                if result.get('url'):
                    content = self.fetch_page(result['url'])
                    # Only keep results where page fetch succeeded
                    if not content.startswith("Error fetching"):
                        result['content'] = content
                        successful_results.append(result)
                    else:
                        # Still include result but mark as fetch failed
                        result['content'] = None
                        successful_results.append(result)
                    
                    # Stop once we have enough successful results
                    if len(successful_results) >= max_results:
                        break
            
            return successful_results[:max_results]
        
        # Return top N results without content fetching
        return results[:max_results]
        
        
    
    def ask(self, prompt: str, system=None, context=None, use_web_search=False, num_search_results=3, explore_pages=False, **kwargs) -> str:
        sys_prompt = system if system is not None else self.system
        
        enhanced_prompt = prompt
        
        # Add provided context if available
        if context:
            enhanced_prompt = f"Context:\n{context}\n\nQuery: {prompt}"
        
        # Perform web search if requested
        if use_web_search and num_search_results > 0:
            # Perform web search and fetch page content given toggle
            search_results = self.web_search(prompt, max_results=num_search_results, fetch_content=explore_pages)
            if search_results:
                # Format search results into context
                search_context = "Web search results:\n"
                for i, result in enumerate(search_results, 1):
                    search_context += f"[{i}] {result['title']}\n"
                    search_context += f"    URL: {result['url']}\n"
                    search_context += f"    {result['snippet']}\n"
                    if explore_pages and result.get('content'):
                        # Include first 500 chars of page content
                        search_context += f"    Page content: {result['content'][:500]}...\n"
                    search_context += "\n"
                # Combine with existing context if any
                if context:
                    enhanced_prompt = f"{enhanced_prompt}\n\n{search_context}"
                else:
                    enhanced_prompt = f"{search_context}\nBased on these search results, {prompt}"
        
        params = {"model": self.model, "input": enhanced_prompt}
        if sys_prompt:
            params["system"] = sys_prompt
        params.update(kwargs)
        r = self.client.responses.create(**params)
        return r.output_text
    
    

