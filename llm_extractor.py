"""
LLM integration for extracting important phrases from PDF pages.
"""

import json
from typing import List, Dict, Optional
from openai import OpenAI


SYSTEM_PROMPT = """You extract ONLY verbatim quotes from the provided page text.
Return JSON. No paraphrases. Your job is to identify the most important phrases that should be highlighted."""

USER_PROMPT_TEMPLATE = """Goal: highlight the most important phrases for case understanding.

Rules:
- Choose 3–7 quotes from this page only.
- Each quote must be copied EXACTLY from the page text (verbatim, no changes).
- 6–25 words per quote (1 sentence max).
- Include page number in each highlight.
- Add a label tag: Problem, Constraint, Numbers, Decision, Risk, Insight, or other relevant category.
- Output JSON: {{"highlights":[{{"page":<page>, "quote":"...", "label":"..."}}]}}

Page: {page_num}
Text:
{page_text}
"""

FULL_CONTEXT_SYSTEM_PROMPT = """You extract ONLY verbatim quotes from the provided full document text.
Return JSON. No paraphrases. Your job is to identify the most important phrases that should be highlighted."""

FULL_CONTEXT_USER_PROMPT_TEMPLATE = """Goal: highlight the most important phrases for case understanding across the entire document.

Rules:
- Choose 15–35 quotes from the full document (unless the document is very short).
- Each quote must be copied EXACTLY from the document text (verbatim, no changes).
- 6–25 words per quote (1 sentence max).
- Include page number in each highlight.
- Add a label tag: Problem, Constraint, Numbers, Decision, Risk, Insight, or other relevant category.
- Output JSON: {{"highlights":[{{"page":<page>, "quote":"...", "label":"..."}}]}}

The document text below includes page markers like:
<<<PAGE 1>>>
...text...
<<<PAGE 2>>>
...text...

Document:
{doc_text}
"""

SUMMARY_SYSTEM_PROMPT = """You summarize business case documents.
Return JSON only. Be concise and structured."""

SUMMARY_USER_PROMPT_TEMPLATE = """Summarize the case document below.

Return JSON with:
- "summary": 5-8 sentences max
- "key_points": 6-10 bullet points
- "open_questions": 3-6 questions a reader should investigate

Document:
{doc_text}
"""

PAGE_EXPLAIN_SYSTEM_PROMPT = """You explain a single page from a case document.
Return plain text. Be concise and helpful."""

PAGE_EXPLAIN_USER_PROMPT_TEMPLATE = """Explain this page in 4-6 sentences.
Focus on what matters for decision making.

Page {page_num}:
{page_text}
"""

def extract_highlights_from_page(
    client: OpenAI, page_num: int, page_text: str, model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Ask LLM to extract important phrases from a single page.

    Args:
        client: OpenAI client instance
        page_num: Page number (1-based)
        page_text: Text content of the page
        model: Model to use (default: gpt-4o-mini)

    Returns:
        List of highlight dicts with "page", "quote", "label" keys
    """
    prompt = USER_PROMPT_TEMPLATE.format(page_num=page_num, page_text=page_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temperature for more consistent verbatim extraction
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        # Validate and return highlights
        highlights = result.get("highlights", [])
        # Ensure all highlights have the correct page number
        for h in highlights:
            h["page"] = page_num

        return highlights

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM on page {page_num}: {e}")
        return []
    except Exception as e:
        print(f"Error calling LLM on page {page_num}: {e}")
        return []


def extract_highlights_from_pdf(
    client: OpenAI,
    pages: List[Dict[str, any]],
    model: str = "gpt-4o-mini",
    max_highlights_per_page: int = 7,
) -> List[Dict]:
    """
    Extract highlights from all pages of a PDF.

    Args:
        client: OpenAI client instance
        pages: List of page dicts with "page" and "text" keys
        model: Model to use
        max_highlights_per_page: Maximum highlights to extract per page

    Returns:
        Combined list of all highlights from all pages
    """
    all_highlights = []

    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"]

        if not page_text.strip():
            print(f"Skipping empty page {page_num}")
            continue

        print(f"Processing page {page_num}...")
        highlights = extract_highlights_from_page(client, page_num, page_text, model)

        # Limit highlights per page
        if len(highlights) > max_highlights_per_page:
            highlights = highlights[:max_highlights_per_page]

        all_highlights.extend(highlights)
        print(f"  Found {len(highlights)} highlights on page {page_num}")

    return all_highlights


def _build_full_doc_text(pages: List[Dict[str, any]]) -> str:
    parts = []
    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"] or ""
        parts.append(f"<<<PAGE {page_num}>>>\n{page_text}\n")
    return "\n".join(parts)


def _summarize_chunk(
    client: OpenAI,
    doc_text: str,
    model: str,
) -> Dict:
    prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(doc_text=doc_text)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    content = response.choices[0].message.content
    return json.loads(content)


def summarize_document(
    client: OpenAI,
    pages: List[Dict[str, any]],
    model: str = "gpt-4o-mini",
    max_context_chars: int = 120000,
) -> Dict:
    """
    Summarize the full document. If too large, summarize chunks then summarize the summaries.
    Returns dict with keys: summary, key_points, open_questions.
    """
    doc_text = _build_full_doc_text(pages)
    if len(doc_text) <= max_context_chars:
        try:
            return _summarize_chunk(client, doc_text, model)
        except Exception as e:
            print(f"Error summarizing full document: {e}")
            return {}

    # Chunk by pages to stay under max_context_chars
    chunks = []
    current = []
    current_len = 0
    for page_info in pages:
        block = f"<<<PAGE {page_info['page']}>>>\n{page_info['text']}\n"
        if current_len + len(block) > max_context_chars and current:
            chunks.append("\n".join(current))
            current = [block]
            current_len = len(block)
        else:
            current.append(block)
            current_len += len(block)
    if current:
        chunks.append("\n".join(current))

    summaries = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            print(f"Summarizing chunk {i}/{len(chunks)}...")
            summaries.append(_summarize_chunk(client, chunk, model))
        except Exception as e:
            print(f"Error summarizing chunk {i}: {e}")

    if not summaries:
        return {}

    # Combine chunk summaries into a final summary
    combined = "\n".join(
        [
            f"Chunk {i+1} summary: {s.get('summary','')}\n"
            f"Key points: {', '.join(s.get('key_points', []))}\n"
            f"Open questions: {', '.join(s.get('open_questions', []))}"
            for i, s in enumerate(summaries)
        ]
    )
    try:
        return _summarize_chunk(client, combined, model)
    except Exception as e:
        print(f"Error summarizing combined chunks: {e}")
        return summaries[0] if summaries else {}


def explain_page(
    client: OpenAI,
    page_num: int,
    page_text: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Explain a single page for a reader.
    """
    prompt = PAGE_EXPLAIN_USER_PROMPT_TEMPLATE.format(
        page_num=page_num, page_text=page_text
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PAGE_EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error explaining page {page_num}: {e}")
        return ""

def extract_highlights_from_pdf_fullcontext(
    client: OpenAI,
    pages: List[Dict[str, any]],
    model: str = "gpt-4o-mini",
    max_context_chars: int = 120000,
    max_highlights_per_page: int = 7,
) -> List[Dict]:
    """
    Extract highlights from the full document context in a single prompt.
    Falls back to per-page extraction if the document is too large.
    """
    doc_text = _build_full_doc_text(pages)
    if len(doc_text) > max_context_chars:
        print(
            f"Full document text is {len(doc_text)} chars; "
            f"exceeds max_context_chars={max_context_chars}. "
            "Falling back to per-page extraction."
        )
        return extract_highlights_from_pdf(
            client,
            pages,
            model=model,
            max_highlights_per_page=max_highlights_per_page,
        )

    prompt = FULL_CONTEXT_USER_PROMPT_TEMPLATE.format(doc_text=doc_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FULL_CONTEXT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        content = response.choices[0].message.content
        result = json.loads(content)
        highlights = result.get("highlights", [])
        return highlights

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from full-context LLM: {e}")
        return []
    except Exception as e:
        print(f"Error calling full-context LLM: {e}")
        return []


def cap_total_highlights(
    client: OpenAI,
    highlights: List[Dict],
    max_total: int = 25,
    model: str = "gpt-4o-mini",
) -> List[Dict]:
    """
    Optionally ask LLM to select the top N highlights from all pages.

    Args:
        client: OpenAI client instance
        highlights: List of all highlights
        max_total: Maximum total highlights to keep
        model: Model to use

    Returns:
        Filtered list of top highlights
    """
    if len(highlights) <= max_total:
        return highlights

    # Create a summary of all highlights for the LLM to rank
    highlights_summary = "\n".join(
        [
            f"Page {h['page']} [{h.get('label', 'N/A')}]: {h['quote'][:100]}..."
            for h in highlights
        ]
    )

    prompt = f"""From the following {len(highlights)} highlights, select the top {max_total} most important ones for case understanding.

Consider:
- Importance for understanding the case
- Diversity across pages
- Key decisions, constraints, numbers, risks

Return JSON with the selected highlights in the same format:
{{"highlights":[{{"page":<page>, "quote":"...", "label":"..."}}]}}

All highlights:
{highlights_summary}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You select the most important highlights from a list.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get("highlights", highlights[:max_total])

    except Exception as e:
        print(f"Error capping highlights, using first {max_total}: {e}")
        return highlights[:max_total]
