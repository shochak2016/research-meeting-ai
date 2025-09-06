import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

KEY = os.environ.get("OPENAI_API_KEY")

'''class designed so that OpenAI object created once in streamlit doc then cached so it does nto have to keep reinstatiating'''
class LLM:
    def __init__(self, openai_api_key, system: Optional[str] = None):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = "gpt-4o-mini"
        self.system = system

    def ask(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        sys_prompt = system if system is not None else self.system
        params = {"model": self.model, "input": prompt}
        if sys_prompt:
            params["system"] = sys_prompt
        params.update(kwargs)
        r = self.client.responses.create(**params)
        return r.output_text
    
    

