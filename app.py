import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from llm_extractor import (
    extract_highlights_from_pdf_fullcontext,
    explain_page,
    summarize_document,
)
from pdf_highlighter import extract_text_per_page, highlight_pdf


load_dotenv()

st.set_page_config(page_title="Case Copilot", layout="wide")

st.title("Case Copilot")
st.caption("Read the case, get guided summaries, and auto-highlights.")


if "pages" not in st.session_state:
    st.session_state.pages = []
if "highlights" not in st.session_state:
    st.session_state.highlights = []
if "summary" not in st.session_state:
    st.session_state.summary = {}
if "page_explanations" not in st.session_state:
    st.session_state.page_explanations = {}
if "manual_highlights" not in st.session_state:
    st.session_state.manual_highlights = []
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "highlighted_pdf_path" not in st.session_state:
    st.session_state.highlighted_pdf_path = None


with st.sidebar:
    st.header("Settings")
    use_env_key = st.checkbox("Use .env API key", value=True)
    api_key_input = st.text_input(
        "OpenAI API key (override)",
        type="password",
        disabled=use_env_key,
        help="If unchecked, provide a key here to override .env",
    )
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    max_context_chars = st.number_input(
        "Max context chars", min_value=20000, max_value=200000, value=120000, step=5000
    )
    render_zoom = st.slider("PDF zoom", min_value=1.0, max_value=3.0, value=2.0, step=0.1)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") if use_env_key else api_key_input
    if not api_key:
        st.warning("Set OPENAI_API_KEY in .env or uncheck and enter it in the sidebar.")
        return None
    return OpenAI(api_key=api_key)


@st.cache_data(show_spinner=False)
def render_page_image(file_path: str, page_num: int, zoom: float) -> bytes:
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    page = doc[page_num - 1]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return pix.tobytes("png")


uploaded = st.file_uploader("Upload a case PDF", type=["pdf"])
if uploaded is not None:
    if st.session_state.pdf_name != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            st.session_state.pdf_path = tmp.name
            st.session_state.pdf_name = uploaded.name
        st.session_state.pages = extract_text_per_page(st.session_state.pdf_path)
        st.session_state.highlights = []
        st.session_state.summary = {}
        st.session_state.page_explanations = {}
        st.session_state.manual_highlights = []
        st.session_state.highlighted_pdf_path = None


if not st.session_state.pdf_path:
    st.stop()

left, right = st.columns([2, 1])

with left:
    st.subheader("Case PDF")
    page_numbers = [p["page"] for p in st.session_state.pages]
    show_highlighted = st.checkbox(
        "Show highlighted PDF",
        value=st.session_state.highlighted_pdf_path is not None,
    )
    pdf_path_to_render = (
        st.session_state.highlighted_pdf_path
        if show_highlighted and st.session_state.highlighted_pdf_path
        else st.session_state.pdf_path
    )
    view_mode = st.radio("View mode", ["Single page", "Multi-page"], horizontal=True)
    if view_mode == "Single page":
        page_choice = st.selectbox("Page", page_numbers, index=0)
        img_bytes = render_page_image(pdf_path_to_render, page_choice, render_zoom)
        st.image(img_bytes, use_container_width=True)
    else:
        for p in page_numbers:
            st.markdown(f"**Page {p}**")
            img_bytes = render_page_image(pdf_path_to_render, p, render_zoom)
            st.image(img_bytes, use_container_width=True)

with right:
    st.subheader("Copilot")

    if st.button("Generate Summary", use_container_width=True):
        client = get_client()
        if client:
            with st.spinner("Summarizing..."):
                st.session_state.summary = summarize_document(
                    client,
                    st.session_state.pages,
                    model=model,
                    max_context_chars=int(max_context_chars),
                )

    if st.session_state.summary:
        st.markdown("**Summary**")
        st.write(st.session_state.summary.get("summary", ""))
        st.markdown("**Key Points**")
        for point in st.session_state.summary.get("key_points", []):
            st.write(f"- {point}")
        st.markdown("**Open Questions**")
        for q in st.session_state.summary.get("open_questions", []):
            st.write(f"- {q}")

    st.divider()

    use_full_context = st.checkbox("Use full-document context", value=True)
    if st.button("Generate Highlights", use_container_width=True):
        client = get_client()
        if client:
            with st.spinner("Extracting highlights..."):
                if use_full_context:
                    highlights = extract_highlights_from_pdf_fullcontext(
                        client,
                        st.session_state.pages,
                        model=model,
                        max_context_chars=int(max_context_chars),
                    )
                else:
                    from llm_extractor import extract_highlights_from_pdf

                    highlights = extract_highlights_from_pdf(
                        client,
                        st.session_state.pages,
                        model=model,
                        max_highlights_per_page=7,
                    )
                for h in highlights:
                    h.setdefault("label", "")
                    h.setdefault("note", "")
                st.session_state.highlights = highlights
                # Auto-apply highlights to generate an annotated PDF for viewing
                if highlights:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as out_tmp:
                        output_path = out_tmp.name
                    highlight_pdf(st.session_state.pdf_path, output_path, highlights)
                    st.session_state.highlighted_pdf_path = output_path

    if st.session_state.highlights:
        st.markdown("**Highlights**")
        for i, h in enumerate(st.session_state.highlights):
            selected_key = f"hl_selected_{i}"
            label_key = f"hl_label_{i}"
            note_key = f"hl_note_{i}"

            if selected_key not in st.session_state:
                st.session_state[selected_key] = True

            st.checkbox(
                f"Page {h['page']}", key=selected_key, value=st.session_state[selected_key]
            )
            st.write(h["quote"])
            st.text_input("Label", key=label_key, value=h.get("label", ""))
            st.text_area("Note", key=note_key, value=h.get("note", ""), height=80)

        if st.button("Apply Highlights to PDF", use_container_width=True):
            selected = []
            for i, h in enumerate(st.session_state.highlights):
                if st.session_state.get(f"hl_selected_{i}"):
                    label = st.session_state.get(f"hl_label_{i}", "").strip()
                    note = st.session_state.get(f"hl_note_{i}", "").strip()
                    content = label
                    if note:
                        content = f"{label} | {note}" if label else note
                    h_out = {
                        "page": h["page"],
                        "quote": h["quote"],
                        "label": content,
                    }
                    selected.append(h_out)

            manual_selected = [
                {
                    "page": mh["page"],
                    "quote": mh["quote"],
                    "label": mh.get("label", ""),
                }
                for mh in st.session_state.manual_highlights
                if mh.get("quote")
            ]

            if selected or manual_selected:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as out_tmp:
                    output_path = out_tmp.name
                all_highlights = selected + manual_selected
                highlight_pdf(st.session_state.pdf_path, output_path, all_highlights)
                st.session_state.highlighted_pdf_path = output_path
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Highlighted PDF",
                        data=f.read(),
                        file_name=f"{os.path.splitext(st.session_state.pdf_name)[0]}_highlighted.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

    st.divider()

    if page_numbers:
        explain_page_choice = st.selectbox("Explain a page", page_numbers)
        if st.button("Explain This Page", use_container_width=True):
            client = get_client()
            if client:
                page_text = st.session_state.pages[explain_page_choice - 1]["text"]
                with st.spinner("Explaining page..."):
                    st.session_state.page_explanations[explain_page_choice] = explain_page(
                        client, explain_page_choice, page_text, model=model
                    )

        if explain_page_choice in st.session_state.page_explanations:
            st.write(st.session_state.page_explanations[explain_page_choice])

    st.divider()

    st.markdown("**Manual Highlight**")
    manual_page = st.selectbox("Manual highlight page", page_numbers, key="manual_page")
    manual_quote = st.text_area("Exact quote from page", key="manual_quote", height=80)
    manual_label = st.text_input("Manual label", key="manual_label")
    if st.button("Add Manual Highlight", use_container_width=True):
        if manual_quote.strip():
            st.session_state.manual_highlights.append(
                {
                    "page": manual_page,
                    "quote": manual_quote.strip(),
                    "label": manual_label.strip(),
                }
            )
        else:
            st.warning("Enter an exact quote from the page to highlight.")
