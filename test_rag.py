#!/usr/bin/env python3
"""
Test RAG script that takes a query from terminal and saves top K results to a text file.
Usage: python test_rag.py "your search query" --top_k 5 --output results.txt
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

def search_papers(query, index_name="pubmed", top_k=5, namespace=None):
    """
    Search for papers using RAG approach
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    
    # Initialize embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(model_name_or_path="nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    
    # Generate query embedding
    print(f"Searching for: {query}")
    query_vector = model.encode(query, normalize_embeddings=True)
    
    # Search in Pinecone
    results = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    
    return results

def format_results(results, query):
    """
    Format search results for text file output
    """
    output = []
    output.append("=" * 80)
    output.append(f"RESEARCH PAPER SEARCH RESULTS")
    output.append(f"Query: {query}")
    output.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Number of results: {len(results['matches'])}")
    output.append("=" * 80)
    output.append("")
    
    for i, match in enumerate(results['matches'], 1):
        metadata = match.get('metadata', {})
        score = match.get('score', 0)
        
        output.append(f"[{i}] RESULT #{i}")
        output.append("-" * 40)
        output.append(f"Relevance Score: {score:.4f}")
        output.append(f"PMID: {metadata.get('pmid', 'N/A')}")
        output.append(f"Title: {metadata.get('title', 'N/A')}")
        
        # Authors
        authors = metadata.get('authors', [])
        if authors:
            if len(authors) > 3:
                authors_str = f"{', '.join(authors[:3])}, et al."
            else:
                authors_str = ', '.join(authors)
            output.append(f"Authors: {authors_str}")
        
        output.append(f"Publication Date: {metadata.get('pub_date', 'N/A')}")
        
        # Abstract
        abstract = metadata.get('abstract', 'N/A')
        if abstract and abstract != 'N/A':
            # Limit abstract length for readability
            if len(abstract) > 500:
                abstract = abstract[:497] + "..."
            output.append(f"\nAbstract:")
            output.append(abstract)
        
        # Link
        pmid = metadata.get('pmid')
        if pmid:
            output.append(f"\nLink: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        
        output.append("")
    
    output.append("=" * 80)
    output.append("END OF RESULTS")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description='Search research papers using RAG')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to retrieve (default: 5)')
    parser.add_argument('--output', type=str, default='rag_results.txt', help='Output file name (default: rag_results.txt)')
    parser.add_argument('--index', type=str, default='pubmed', help='Pinecone index name (default: pubmed)')
    parser.add_argument('--namespace', type=str, default=None, help='Pinecone namespace (optional)')
    parser.add_argument('--json', action='store_true', help='Also save results as JSON')
    
    args = parser.parse_args()
    
    try:
        # Perform search
        results = search_papers(
            query=args.query,
            index_name=args.index,
            top_k=args.top_k,
            namespace=args.namespace
        )
        
        if not results['matches']:
            print("No results found!")
            sys.exit(1)
        
        # Format and save results
        formatted_output = format_results(results, args.query)
        
        # Print results to terminal
        print("\n" + "="*80)
        print("SEARCH RESULTS (also saving to file...)")
        print("="*80)
        
        for i, match in enumerate(results['matches'], 1):
            metadata = match.get('metadata', {})
            score = match.get('score', 0)
            
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"    Title: {metadata.get('title', 'N/A')}")
            
            # Authors
            authors = metadata.get('authors', [])
            if authors:
                if len(authors) > 3:
                    authors_str = f"{', '.join(authors[:3])}, et al."
                else:
                    authors_str = ', '.join(authors)
                print(f"    Authors: {authors_str}")
            
            print(f"    Date: {metadata.get('pub_date', 'N/A')}")
            print(f"    PMID: {metadata.get('pmid', 'N/A')}")
            
            # Abstract preview
            abstract = metadata.get('abstract', '')
            if abstract:
                preview = abstract[:200] + "..." if len(abstract) > 200 else abstract
                print(f"    Abstract: {preview}")
            
            # Link
            pmid = metadata.get('pmid')
            if pmid:
                print(f"    Link: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        
        print("\n" + "="*80)
        
        # Save to text file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        print(f"\n✅ Full results saved to: {args.output}")
        
        # Optionally save JSON
        if args.json:
            json_filename = args.output.replace('.txt', '.json')
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': args.query,
                    'timestamp': datetime.now().isoformat(),
                    'top_k': args.top_k,
                    'results': results['matches']
                }, f, indent=2)
            print(f"✅ JSON results saved to: {json_filename}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"  - Query: '{args.query}'")
        print(f"  - Found: {len(results['matches'])} papers")
        print(f"  - Best match score: {results['matches'][0]['score']:.4f}")
        print(f"  - Output file: {args.output}")
        
    except KeyError as e:
        print(f"❌ Error: Missing environment variable. Make sure PINECONE_API_KEY is set in .env")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()