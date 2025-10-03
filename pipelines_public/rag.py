from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os


from typing import Any, Optional, List

from pinecone import Pinecone

from operations.embedding import Embedder

OPENAI_API_KEY = "OPENAI_API_KEY"
PINECONE_API_KEY = "PINECONE_API_KEY"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) #temp set to zero, would prefer less distribution (less chance for error)

class FindSimilar:
    def __init__(self, idx: Any, top_k: int = 3, flt: Optional[Any] = None, 
                 namespace: Optional[str] = None):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.idx = idx
        self.k = top_k
        self.namespace = namespace
        self.flt = flt
        self.embedder = Embedder(obj=None)

    def encode_query(self, query):
        vec = self.embedder.str_to_vec(text=query, is_query=True)
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return vec

    def find_similar(self, q):
        import time
        start_time = time.time()

        qvec = self.encode_query(q)

        res = self.idx.query(
            vector=qvec,
            top_k=self.k,
            include_metadata=True,
            namespace=self.namespace,
            filter=self.flt
        )

        matches = getattr(res, "matches", None) or res.get("matches", [])

        docs = []
        for m in matches:
            md = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else {}) or {}
            mid = getattr(m, "id", None) if not isinstance(m, dict) else m.get("id")
            mscore = getattr(m, "score", None) if not isinstance(m, dict) else m.get("score")

            # Build metadata dict
            meta = {**md, "_id": mid, "_score": mscore}

            # If you don’t need text, make page_content minimal
            docs.append(Document(page_content="", metadata=meta))

        elapsed_time = time.time() - start_time
        return docs, elapsed_time

def build_rag(retriever, model="gpt-4o-mini", temperature=0.0, per_field_chars=1000, return_abstract=False):
    '''
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    index = pc.Index(index_name)
    retriever = FindSimilar(query=query, idx=index)
    '''
    def format_docs(docs):
        blocks = []
        for i, d in enumerate(docs, 1):
            md = d.metadata
            title = (md.get("title") or md.get("name") or "")
            url = (md.get("link") or md.get("url") or md.get("source") or "")
            abstract = md.get("abstract") or ""
            text = (d.page_content or "")
            if per_field_chars is not None:
                title = title[:per_field_chars]
                if abstract:
                    abstract = abstract[:per_field_chars]
                text = text[:per_field_chars]
            block = [f"[{i}]"]
            if title:
                block.append(f"TITLE: {title}")
            if url:
                block.append(f"URL: {url}")
            # Use abstract from metadata if available, otherwise fall back to page_content
            if abstract:
                block.append(f"ABSTRACT: {abstract}")
            elif text:
                block.append(f"ABSTRACT: {text}")
            blocks.append("\n".join(block))
        return "\n\n---\n\n".join(blocks)

    if return_abstract == False:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. You will be given N retrieved documents as CONTEXT. "
                "For EACH document, produce a JSON object with keys: title (string), summary (2-4 sentences), link (string or null). "
                "Use TITLE and URL from the context when present. Base the summary strictly on ABSTRACT. "
                "Return ONLY a JSON array of objects, in the same order as the context blocks [1], [2], ...; no extra text.",
            ),
            ("human", "CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"),
        ])

    else:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. You will be given N retrieved documents as CONTEXT. "
                "For EACH document, produce a JSON object with keys: title (string), abstract (full text), summary (2-4 sentences), link (string or null). "
                "Use TITLE and URL from the context when present. Base the summary strictly on ABSTRACT. "
                "Return ONLY a JSON array of objects, in the same order as the context blocks [1], [2], ...; no extra text.",
            ),
            ("human", "CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"),
        ])

    llm = ChatOpenAI(model=model, temperature=temperature)
    parser = JsonOutputParser()

    # Build chain that expects docs to be provided 
    chain = (
        {"context": (lambda x: format_docs(x["docs"])), "question": (lambda x: x["question"]) }
        | prompt
        | llm
        | parser
    )

    def ask(q):
        docs = retriever.invoke(q)
        items = chain.invoke({"question": q, "docs": docs})
        for i, (it, d) in enumerate(zip(items, docs), 1):
            it["_ref"] = d.metadata.get("link") or d.metadata.get("url") or d.metadata.get("source") or d.metadata.get("_id")
            it["_score"] = d.metadata.get("_score")
        return {"results": items, "documents": docs}

    return ask
