#!/usr/bin/env python3
"""
Main entry point for the PDF highlighter.
"""
import argparse
import os

from dotenv import load_dotenv

load_dotenv()  # load OPENAI_API_KEY from .env if present

from openai import OpenAI
from pdf_highlighter import extract_text_per_page, highlight_pdf
from llm_extractor import (
    extract_highlights_from_pdf,
    extract_highlights_from_pdf_fullcontext,
    cap_total_highlights,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract important phrases from PDF using LLM and highlight them"
    )
    parser.add_argument("input_pdf", help="Path to input PDF file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output PDF file (default: input_pdf with '_highlighted' suffix)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY in .env or env)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-per-page",
        type=int,
        default=7,
        help="Maximum highlights per page (default: 7)"
    )
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Use full-document context in a single LLM call (if size permits)"
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=120000,
        help="Max characters for full-context prompt before fallback (default: 120000)"
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Maximum total highlights across all pages (default: no limit)"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM extraction (for testing with existing highlights JSON)"
    )
    parser.add_argument(
        "--highlights-json",
        default=None,
        help="Path to JSON file with highlights (if skipping LLM)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input PDF not found: {args.input_pdf}")
        return 1
    
    # Determine output path
    if args.output:
        output_pdf = args.output
    else:
        base, ext = os.path.splitext(args.input_pdf)
        output_pdf = f"{base}_highlighted{ext}"
    
    # Extract text from PDF
    print(f"Extracting text from PDF: {args.input_pdf}")
    pages = extract_text_per_page(args.input_pdf)
    print(f"Found {len(pages)} pages")
    
    # Get highlights
    if args.skip_llm:
        # Load from JSON file
        if not args.highlights_json:
            print("Error: --highlights-json required when using --skip-llm")
            return 1
        
        import json
        with open(args.highlights_json, 'r') as f:
            data = json.load(f)
            highlights = data.get("highlights", [])
        print(f"Loaded {len(highlights)} highlights from {args.highlights_json}")
    else:
        # Use LLM to extract highlights
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required. Put OPENAI_API_KEY=your-key in .env or use --api-key")
            return 1
        
        client = OpenAI(api_key=api_key)
        
        print(f"Extracting highlights using {args.model}...")
        if args.full_context:
            highlights = extract_highlights_from_pdf_fullcontext(
                client,
                pages,
                model=args.model,
                max_context_chars=args.max_context_chars,
                max_highlights_per_page=args.max_per_page,
            )
        else:
            highlights = extract_highlights_from_pdf(
                client,
                pages,
                model=args.model,
                max_highlights_per_page=args.max_per_page
            )
        
        # Optionally cap total highlights
        if args.max_total and len(highlights) > args.max_total:
            print(f"Capping highlights to top {args.max_total}...")
            highlights = cap_total_highlights(
                client,
                highlights,
                max_total=args.max_total,
                model=args.model
            )
    
    print(f"\nTotal highlights to apply: {len(highlights)}")
    
    # Apply highlights to PDF
    print(f"\nHighlighting PDF...")
    highlight_pdf(args.input_pdf, output_pdf, highlights)
    
    print(f"\nDone! Highlighted PDF saved to: {output_pdf}")
    return 0


if __name__ == "__main__":
    exit(main())
