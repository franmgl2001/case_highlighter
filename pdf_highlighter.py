"""
Core PDF highlighting module with robust phrase matching.
"""
import re
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from rapidfuzz import fuzz, process


def normalize(s: str) -> str:
    """
    Normalize text for matching: remove soft hyphens, line-break hyphens, collapse whitespace.
    """
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"-\s*\n\s*", "", s)  # remove hyphenation across line breaks
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def find_rects_for_quote(page: fitz.Page, quote: str) -> List[fitz.Rect]:
    """
    Returns a list of rectangles to highlight for the quote on this page.
    Tries exact search first, then chunk search.
    """
    q = normalize(quote)

    # 1) exact-ish search using PyMuPDF built-in (works on raw page text layout)
    rects = page.search_for(quote, quads=False)  # try raw quote first
    if rects:
        return rects

    # try normalized quote (sometimes helps)
    rects = page.search_for(q, quads=False)
    if rects:
        return rects

    # 2) chunk fallback: split into word windows
    words = q.split()
    if len(words) < 6:
        return []

    windows = []
    window_size = min(10, len(words))
    step = max(3, window_size // 2)

    for i in range(0, len(words) - window_size + 1, step):
        chunk = " ".join(words[i:i+window_size])
        windows.append(chunk)

    found = []
    for chunk in windows:
        r = page.search_for(chunk, quads=False)
        if r:
            found.extend(r)

    # if we found at least something, return it
    if found:
        return found

    return []


def fuzzy_best_line(page_text: str, quote: str) -> Optional[str]:
    """
    If exact search fails, try matching the quote to the closest line.
    Returns best matching line (string) or None.
    """
    q = normalize(quote)
    lines = [normalize(l) for l in page_text.split("\n") if normalize(l)]
    if not lines:
        return None

    best = process.extractOne(q, lines, scorer=fuzz.partial_ratio)
    if best and best[1] >= 85:  # threshold you can tune
        return best[0]
    return None


def highlight_pdf(input_pdf: str, output_pdf: str, highlights: List[Dict]) -> None:
    """
    Highlights phrases in a PDF based on the provided highlights list.
    
    Args:
        input_pdf: Path to input PDF file
        output_pdf: Path to output PDF file (will be created/overwritten)
        highlights: List of dicts with keys:
            - "page": int (1-based page number)
            - "quote": str (verbatim quote to highlight)
            - "label": str (optional, e.g., "Constraint", "Numbers", "Decision")
    """
    doc = fitz.open(input_pdf)

    for h in highlights:
        page_num = h["page"] - 1  # convert to 0-based
        quote = h["quote"]

        if page_num < 0 or page_num >= doc.page_count:
            print(f"Warning: Page {h['page']} out of range (1-{doc.page_count}), skipping quote: {quote[:50]}...")
            continue

        page = doc[page_num]

        rects = find_rects_for_quote(page, quote)

        # 3) fuzzy fallback: match closest line, then search_for that line
        if not rects:
            page_text = page.get_text("text")
            best_line = fuzzy_best_line(page_text, quote)
            if best_line:
                rects = page.search_for(best_line, quads=False)
                if rects:
                    print(f"Found quote via fuzzy match on page {h['page']}")

        if not rects:
            print(f"Warning: Could not find quote on page {h['page']}: {quote[:50]}...")
            continue

        for rect in rects:
            annot = page.add_highlight_annot(rect)
            # optional: store label/comment in the annotation
            if "label" in h and h["label"]:
                annot.set_info(content=h["label"])
            annot.update()

    doc.save(output_pdf, deflate=True)
    doc.close()
    print(f"Saved highlighted PDF to: {output_pdf}")


def extract_text_per_page(pdf_path: str) -> List[Dict[str, any]]:
    """
    Extract text from each page of the PDF.
    
    Returns:
        List of dicts with "page" (1-based) and "text" keys
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append({
            "page": page_num + 1,  # 1-based
            "text": text
        })
    
    doc.close()
    return pages

