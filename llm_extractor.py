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
