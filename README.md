# PDF Case Highlighter

An intelligent PDF highlighter that uses LLMs to identify and highlight important phrases in case study PDFs.

## How It Works

1. **Extract text per page** from the PDF
2. **Ask the LLM** for important verbatim quotes (6-25 words) with page numbers
3. **Find those phrases** on the right page using robust matching (exact → chunk → fuzzy)
4. **Highlight them** in the PDF and return the annotated file

## End-to-End Workflow

This is the full app flow from CLI to output:

1. **CLI entry**: `main.py` parses arguments, loads `.env`, validates the input PDF, and computes the output path.
2. **Text extraction**: `extract_text_per_page()` in `pdf_highlighter.py` uses PyMuPDF to extract raw text per page into a list of `{page, text}` objects.
3. **Highlight source (two paths)**. LLM path (default): `extract_highlights_from_pdf()` iterates pages and calls `extract_highlights_from_page()` for each page. Each page prompt asks for 3–7 verbatim quotes (6–25 words) and a label. Results are normalized so each highlight has the correct 1-based page number. If `--full-context` is set, the app sends the entire document (with page markers) in a single LLM call via `extract_highlights_from_pdf_fullcontext()` and asks for quotes across the whole case. If `--max-total` is set, `cap_total_highlights()` optionally re-ranks across pages and keeps the top N. Skip-LLM path: `--skip-llm --highlights-json <file>` loads a prebuilt JSON list of `{page, quote, label}` objects.
4. **Highlighting**: `highlight_pdf()` iterates the highlights and tries to locate each quote on its page using:
   - Exact search (raw quote and normalized quote)
   - Chunk search (sliding word windows across the quote)
   - Fuzzy match (best line match, then search on that line)
5. **Annotate and save**: Highlight annotations are added to the PDF and the result is written to `*_highlighted.pdf`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Put your OpenAI API key in a `.env` file in the project folder:

```
OPENAI_API_KEY=sk-your-key-here
```

Then run:

```bash
python main.py input.pdf
```

This will create `input_highlighted.pdf` with highlights. You can also set `OPENAI_API_KEY` in your shell or pass `--api-key` on the command line.

### Advanced Options

```bash
python main.py input.pdf \
  --output output.pdf \
  --model gpt-4o \
  --max-per-page 5 \
  --max-total 25
```

### Testing Without LLM

You can test the highlighting logic with a pre-made highlights JSON:

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
# case_highlighter
