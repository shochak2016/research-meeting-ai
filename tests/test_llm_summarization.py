#!/usr/bin/env python3
"""
Test script for LLM summarization using the ask function from pipelines_public/rag.py
Usage: python test_llm_summarization.py "your search query" --top_k 3 --return_abstract
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines.rag import ask

# Load environment variables
load_dotenv()

# Initialize rich console for pretty printing
console = Console()

def print_results(result, query, show_abstract=False):
    """
    Pretty print the LLM summarization results to terminal
    """
    console.print("\n" + "="*80, style="bold blue")
    console.print(f"🔍 LLM SUMMARIZATION RESULTS", style="bold green")
    console.print(f"Query: '{query}'", style="italic")
    console.print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
    console.print("="*80 + "\n", style="bold blue")
    
    # Term Summary Section
    if result.get("term_summary"):
        summary_panel = Panel(
            result["term_summary"],
            title="📚 Term Summary",
            title_align="left",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(summary_panel)
        console.print()
    
    # Latest Findings Section
    if result.get("latest_findings"):
        findings_panel = Panel(
            result["latest_findings"],
            title="🔬 Latest Literature Findings",
            title_align="left",
            border_style="green",
            padding=(1, 2)
        )
        console.print(findings_panel)
        console.print()
    
    # References Table
    references = result.get("references", [])
    if references:
        console.print("📖 [bold]References Found:[/bold]", style="yellow")
        
        table = Table(show_header=True, header_style="bold magenta", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan", max_width=50)
        table.add_column("Score", justify="right", style="green")
        table.add_column("Link", style="blue", max_width=30)
        
        for i, ref in enumerate(references, 1):
            title = ref.get("title", "N/A")
            if len(title) > 47:
                title = title[:44] + "..."
            score = f"{ref.get('score', 0):.4f}" if ref.get('score') else "N/A"
            link = ref.get("link", "")
            if link and len(link) > 27:
                link = "..." + link[-24:]
            
            table.add_row(str(i), title, score, link)
        
        console.print(table)
        console.print()
    
    # Document Details (if verbose/abstract mode)
    if show_abstract and result.get("documents"):
        console.print("📄 [bold]Document Details:[/bold]", style="yellow")
        
        for i, doc in enumerate(result["documents"], 1):
            metadata = doc.metadata
            
            doc_text = Text()
            doc_text.append(f"\n[{i}] ", style="bold")
            doc_text.append(metadata.get("title", "N/A"), style="cyan")
            
            details = []
            
            # Authors
            authors = metadata.get("authors", [])
            if authors:
                if len(authors) > 3:
                    authors_str = f"{', '.join(authors[:3])}, et al."
                else:
                    authors_str = ', '.join(authors)
                details.append(f"Authors: {authors_str}")
            
            # Date
            if metadata.get("pub_date"):
                details.append(f"Date: {metadata.get('pub_date')}")
            
            # PMID
            if metadata.get("pmid"):
                details.append(f"PMID: {metadata.get('pmid')}")
            
            # Abstract
            if show_abstract and metadata.get("abstract"):
                abstract = metadata.get("abstract")
                if len(abstract) > 300:
                    abstract = abstract[:297] + "..."
                details.append(f"\nAbstract:\n{abstract}")
            
            details_panel = Panel(
                "\n".join(details),
                title=doc_text,
                title_align="left",
                border_style="dim",
                padding=(1, 2)
            )
            console.print(details_panel)
    
    # Statistics
    console.print("\n" + "="*80, style="bold blue")
    console.print("📊 [bold]Statistics:[/bold]", style="yellow")
    console.print(f"  • Documents retrieved: {len(result.get('documents', []))}")
    console.print(f"  • References generated: {len(result.get('references', []))}")
    if references and references[0].get('score'):
        console.print(f"  • Best match score: {references[0]['score']:.4f}")
    console.print("="*80 + "\n", style="bold blue")

def main():
    parser = argparse.ArgumentParser(description='Test LLM summarization with the ask function')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--top_k', type=int, default=3, help='Number of documents to retrieve (default: 3)')
    parser.add_argument('--index', type=str, default='pubmed', help='Pinecone index (default: pubmed)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model (default: gpt-4o-mini)')
    parser.add_argument('--return_abstract', action='store_true', help='Include abstracts in LLM response')
    parser.add_argument('--namespace', type=str, default=None, help='Pinecone namespace')
    parser.add_argument('--show_docs', action='store_true', help='Show document details in output')
    parser.add_argument('--save', type=str, help='Save results to file')
    
    args = parser.parse_args()
    
    try:
        console.print(f"\n🚀 [bold]Initializing LLM Summarization Pipeline[/bold]", style="cyan")
        console.print(f"  • Index: {args.index}")
        console.print(f"  • Model: {args.model}")
        console.print(f"  • Top K: {args.top_k}")
        console.print(f"  • Return abstracts: {args.return_abstract}")
        console.print(f"  • Namespace: {args.namespace or 'default'}")
        
        console.print(f"\n🔎 Searching for: [bold]'{args.query}'[/bold]", style="green")
        console.print("⏳ Processing query (this may take a moment)...\n")
        
        # Call the ask function
        with console.status("[bold green]Running vector search and LLM processing...") as status:
            result = ask(
                query=args.query,
                index_name=args.index,
                top_k=args.top_k,
                model=args.model,
                return_abstract=args.return_abstract,
                namespace=args.namespace
            )
        
        # Print results
        print_results(result, args.query, show_abstract=args.show_docs or args.return_abstract)
        
        # Save to file if requested
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                f.write(f"Query: {args.query}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Top K: {args.top_k}\n\n")
                
                f.write("TERM SUMMARY:\n")
                f.write(result.get("term_summary", "N/A") + "\n\n")
                
                f.write("LATEST FINDINGS:\n")
                f.write(result.get("latest_findings", "N/A") + "\n\n")
                
                f.write("REFERENCES:\n")
                for i, ref in enumerate(result.get("references", []), 1):
                    f.write(f"{i}. {ref.get('title', 'N/A')}\n")
                    f.write(f"   Link: {ref.get('link', 'N/A')}\n")
                    f.write(f"   Score: {ref.get('score', 'N/A')}\n\n")
            
            console.print(f"✅ Results saved to: {args.save}", style="green")
        
    except KeyError as e:
        console.print(f"❌ [bold red]Error:[/bold red] Missing environment variable", style="red")
        console.print(f"   Make sure PINECONE_API_KEY and OPENAI_API_KEY are set in .env", style="dim")
        console.print(f"   Missing: {e}", style="dim")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {str(e)}", style="red")
        import traceback
        if "--debug" in sys.argv:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()