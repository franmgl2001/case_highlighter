# PDF Case Highlighter + Copilot

A simple toolkit for highlighting case PDFs and a Streamlit copilot app for reading, summarizing, and annotating cases.

## What You Can Do

- Highlight important quotes in PDFs with LLMs
- Generate summaries and key points
- Explain specific pages while you read
- Add manual highlights and notes

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add your OpenAI API key in a `.env` file at the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

## Use the App (Recommended)

Run the Streamlit copilot UI:

```bash
streamlit run app.py
```

What’s inside:
- PDF reader with single-page and multi-page modes
- Full-document summary
- Auto-highlights (full-document context toggle)
- Manual highlights with labels and notes
- Page explanations on demand
- Download annotated PDF

By default the app reads `OPENAI_API_KEY` from `.env`. You can override it in the sidebar if needed.

## Use the Script Only (CLI)

Basic usage:

```bash
python main.py input.pdf
```

This creates `input_highlighted.pdf` with highlights.

### Full-Document Context

Use full-document context to improve quote selection:

```bash
python main.py input.pdf --full-context
```

### Advanced Options

```bash
python main.py input.pdf \
  --output output.pdf \
  --model gpt-4o \
  --max-per-page 5 \
  --max-total 25
```

### Testing Without LLM

Provide a predefined highlights JSON:

```json
{
  "highlights": [
    {
      "page": 3,
      "quote": "The plant must reduce lead time from 12 days to 5 days within 6 months.",
      "label": "Constraint"
    },
    {
      "page": 7,
      "quote": "Gross margin dropped from 38% to 29% due to higher raw material costs.",
      "label": "Numbers"
    }
  ]
}
```

```bash
python main.py input.pdf --skip-llm --highlights-json highlights.json
```

## Command-Line Arguments

- `input_pdf`: Path to input PDF file (required)
- `-o, --output`: Output PDF path (default: `{input}_highlighted.pdf`)
- `--api-key`: OpenAI API key (or put `OPENAI_API_KEY` in `.env`)
- `--model`: Model to use (default: `gpt-4o-mini`)
- `--max-per-page`: Max highlights per page (default: 7)
- `--max-total`: Max total highlights across all pages (optional)
- `--full-context`: Use full-document context in a single LLM call (if size permits)
- `--max-context-chars`: Max characters for full-context prompt before fallback (default: 120000)
- `--skip-llm`: Skip LLM extraction (use with `--highlights-json`)
- `--highlights-json`: Path to JSON file with highlights

## How It Works (Pipeline)

1. **Extract text per page** from the PDF
2. **Ask the LLM** for important verbatim quotes (6–25 words) with page numbers
3. **Find those phrases** on the right page using robust matching (exact → chunk → fuzzy)
4. **Highlight them** in the PDF and return the annotated file

## Matching Strategy

The highlighter uses a three-layer matching strategy:

1. **Exact search**: Try to find the quote verbatim
2. **Chunk search**: Split quote into word windows and search for chunks
3. **Fuzzy match**: Match to the closest line using similarity scoring

This handles common PDF issues like:
- Line breaks in the middle of sentences
- Hyphenated line wraps (e.g., `inter-\nnational`)
- Weird whitespace

## Highlight Labels

The LLM assigns labels to highlights:
- `Problem`
- `Constraint`
- `Numbers`
- `Decision`
- `Risk`
- `Insight`
- Other relevant categories

Labels are stored in the PDF annotation metadata.

## Requirements

- Python 3.8+
- PyMuPDF (fitz) for PDF operations
- rapidfuzz for fuzzy matching
- OpenAI Python client for LLM integration
- Streamlit for the copilot app
