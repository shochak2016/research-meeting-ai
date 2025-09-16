from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pinecone import Pinecone

from operations.embedding import Embedder

OPENAI_API_KEY = "OPENAI_API_KEY"
PINECONE_API_KEY = "PINECONE_API_KEY"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) #temp set to zero, would prefer less distribution (less chance for error)

class FindSimilar(BaseRetriever):
    def __init__(self, query, idx, top_k=3, flt=None, namespace=None, key_content="abstract"):
        self.idx = idx
        self.k = top_k
        self.namespace=namespace
        self.flt = flt
        self.query = query
        self.embedder = Embedder(obj=None)
        self.key_content = key_content

    def encode_query(self):       
        vec = self.embedder.str_to_vec(text=self.query, is_query=True)
        return vec

    def find_similar(self):
        qvec = self.encode_query()
        res = self.index.query(vector=qvec, top_k=self.k, include_metadata=True, namespace=self.namespace, filter=self.flt)
        matches = res.get("matches", []) #only keeps list of relevant "matches" values from res dict
        docs = []
        for m in res.get("matches", []):
            md = m.get("metadata") or {}
            text = md.get(self.text_key)
            if not text:
                continue
            link = md.get("link") or md.get("url") or md.get("source") or (
                f"https://pubmed.ncbi.nlm.nih.gov/{md.get('pmid')}/" if md.get("pmid") else ""
            )
            meta = {**md, "_id": m.get("id"), "_score": m.get("score")}
            if link:
                meta["link"] = link
            docs.append(Document(page_content=text, metadata=meta))
        return docs

def build_rag(query, index_name, model="gpt-4o-mini", temperature=0.0, per_field_chars=1000):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    retriever = FindSimilar(query=query, idx=index)

    def format_docs(docs):
        blocks = []
        for i, d in enumerate(docs, 1):
            md = d.metadata
            title = (md.get("title") or md.get("name") or "")
            url = (md.get("link") or md.get("url") or md.get("source") or "")
            text = (d.page_content or "")
            if per_field_chars is not None:
                title = title[:per_field_chars]
                text = text[:per_field_chars]
            block = [f"[{i}]"]
            if title:
                block.append(f"TITLE: {title}")
            if url:
                block.append(f"URL: {url}")
            if text:
                block.append(f"ABSTRACT: {text}")
            blocks.append("\n".join(block))
        return "\n\n---\n\n".join(blocks)

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
