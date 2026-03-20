from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


ROOT = Path(__file__).resolve().parent
DICIONARIO_PDF_PATH = ROOT / "doc" / "PEDE_ Pontos importantes.pdf"


@st.cache_data(show_spinner=False)
def _load_pdf_bytes(pdf_path: str) -> bytes:
    return Path(pdf_path).read_bytes()


@st.cache_data(show_spinner=False)
def _render_pdf_pages(pdf_path: str, zoom: float = 2.0) -> list[bytes]:
    if fitz is None:
        return []

    rendered_pages: list[bytes] = []
    with fitz.open(pdf_path) as doc:
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            rendered_pages.append(pixmap.tobytes("png"))
    return rendered_pages


def render_dicionario_tab() -> None:
    st.subheader("Dicionário")
    st.caption("Versão Streamlit do documento de referência: PEDE - Pontos importantes.")

    if not DICIONARIO_PDF_PATH.exists():
        st.error(f"PDF não encontrado em: {DICIONARIO_PDF_PATH}")
        return

    try:
        pdf_bytes = _load_pdf_bytes(str(DICIONARIO_PDF_PATH))
    except Exception as error:
        st.error(f"Falha ao carregar o PDF do dicionário: {error}")
        return

    st.download_button(
        label="Baixar PDF do Dicionário",
        data=pdf_bytes,
        file_name=DICIONARIO_PDF_PATH.name,
        mime="application/pdf",
        width="stretch",
    )

    if fitz is None:
        st.error(
            "A biblioteca PyMuPDF não está disponível para renderizar as páginas no Streamlit. "
            "Instale com: pip install pymupdf"
        )
        return

    with st.spinner("Renderizando o dicionário no Streamlit..."):
        pages = _render_pdf_pages(str(DICIONARIO_PDF_PATH), zoom=1.4)

    if not pages:
        st.error("Não foi possível gerar as páginas do documento.")
        return

    st.markdown(f"**Total de páginas:** {len(pages)}")

    for page_image in pages:
        _, col_center, _ = st.columns([1, 8, 1])
        with col_center:
            st.image(
                page_image,
                width="stretch",
            )
