import os
from dotenv import load_dotenv
from pinecone import Pinecone

from pipelines.rag2 import FindSimilar

def main():
    load_dotenv()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("pubmed")

    obj = FindSimilar(idx=index)

    docs, latency = obj.find_similar("CRISPR gene editing in cats")

    print(f"latency: {latency:.3f} seconds")

    if not docs:
        print("No matches found.")
        return

    for d in docs:
        print("\n--- Document ---")
        print(f"title: {d.metadata.get('title', 'NONE')}")
        print(f"authors: {d.metadata.get('authors', 'NONE')}")
        print(f"link: {d.metadata.get('link', 'NONE')}")
        print(f"id: {d.metadata.get('_id', 'NONE')}")
        print(f"score: {d.metadata.get('_score', 'NONE')}")

if __name__ == "__main__":
    main()
