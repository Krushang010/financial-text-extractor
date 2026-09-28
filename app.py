import streamlit as st
from pypdf import PdfReader

from src.extractor import (
    extract_financial_data,
    financial_to_dataframe,
    developments_to_dataframe
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Financial Text Extractor",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📊 Financial Text Extractor")

st.write(
    "Analyze financial news, earnings commentary, "
    "company updates or complete earnings transcripts "
    "using an LLM."
)


# ---------------------------------------------------------
# INPUT METHOD
# ---------------------------------------------------------

input_method = st.radio(
    "Choose Input Method",
    [
        "Paste Text",
        "Upload PDF / TXT"
    ],
    horizontal=True
)


article_text = ""


# ---------------------------------------------------------
# OPTION 1 — PASTE TEXT
# ---------------------------------------------------------

if input_method == "Paste Text":

    article_text = st.text_area(
        "Financial Text",
        height=450,
        placeholder=(
            "Paste financial news, earnings commentary "
            "or transcript text here..."
        )
    )


    if article_text:

        st.caption(
            f"{len(article_text):,} characters | "
            f"{len(article_text.split()):,} words"
        )


# ---------------------------------------------------------
# OPTION 2 — FILE UPLOAD
# ---------------------------------------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload Financial Transcript",
        type=[
            "pdf",
            "txt"
        ]
    )


    if uploaded_file is not None:

        # -------------------------------------------------
        # PDF
        # -------------------------------------------------

        if uploaded_file.type == "application/pdf":

            try:

                reader = PdfReader(
                    uploaded_file
                )


                pages = []


                for page in reader.pages:

                    page_text = (
                        page.extract_text()
                    )


                    if page_text:

                        pages.append(
                            page_text
                        )


                article_text = "\n\n".join(
                    pages
                )


                if article_text.strip():

                    st.success(
                        "PDF loaded successfully."
                    )


                else:

                    st.warning(
                        "No readable text was found in the PDF. "
                        "The PDF may be scanned or image-based."
                    )


            except Exception as e:

                st.error(
                    f"Unable to read PDF: {e}"
                )


        # -------------------------------------------------
        # TXT
        # -------------------------------------------------

        else:

            try:

                article_text = (
                    uploaded_file
                    .read()
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                )


                st.success(
                    "Text file loaded successfully."
                )


            except Exception as e:

                st.error(
                    f"Unable to read text file: {e}"
                )


        # -------------------------------------------------
        # FILE STATISTICS
        # -------------------------------------------------

        if article_text:

            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Characters",
                    f"{len(article_text):,}"
                )


            with col2:

                st.metric(
                    "Words",
                    f"{len(article_text.split()):,}"
                )


# ---------------------------------------------------------
# EXTRACT BUTTON
# ---------------------------------------------------------

if st.button(
    "Extract Financial Insights",
    type="primary"
):

    if not article_text.strip():

        st.warning(
            "Please paste financial text "
            "or upload a file first."
        )


    else:

        # -------------------------------------------------
        # INFO
        # -------------------------------------------------

        st.info(
            "Long transcripts are automatically divided "
            "into smaller chunks. Because this project uses "
            "the Groq free tier, large transcripts may take "
            "a few minutes to process."
        )


        # -------------------------------------------------
        # PROGRESS BAR
        # -------------------------------------------------

        progress_bar = st.progress(
            0,
            text="Preparing transcript..."
        )


        def update_progress(
            current,
            total
        ):

            percentage = (
                current / total
            )


            progress_bar.progress(
                percentage,
                text=(
                    f"Analyzing transcript — "
                    f"chunk {current} of {total}"
                )
            )


        # -------------------------------------------------
        # LLM EXTRACTION
        # -------------------------------------------------

        result = extract_financial_data(
            article_text,
            progress_callback=update_progress
        )


        progress_bar.empty()


        # -------------------------------------------------
        # ERROR
        # -------------------------------------------------

        if "error" in result:

            st.error(
                result["error"]
            )


            if "details" in result:

                st.caption(
                    result["details"]
                )


        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------

        else:

            st.success(
                "Extraction completed."
            )


            # -------------------------------------------------
            # TRANSCRIPT SIGNALS
            # -------------------------------------------------

            st.subheader(
                "Transcript Signals"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Congratulations Count",
                    result.get(
                        "congratulations_count",
                        0
                    )
                )


            with col2:

                st.metric(
                    "Chunks Processed",
                    result.get(
                        "chunks_processed",
                        1
                    )
                )


            with col3:

                st.metric(
                    "Chunks Skipped",
                    result.get(
                        "chunks_skipped",
                        0
                    )
                )


            # -------------------------------------------------
            # FINANCIAL INFORMATION
            # -------------------------------------------------

            st.subheader(
                "Financial Information"
            )


            financial_df = (
                financial_to_dataframe(
                    result
                )
            )


            st.dataframe(
                financial_df,
                hide_index=True,
                use_container_width=True
            )


            # -------------------------------------------------
            # WHAT'S NEW
            # -------------------------------------------------

            st.subheader(
                "🆕 What's New"
            )


            developments_df = (
                developments_to_dataframe(
                    result
                )
            )


            if developments_df.empty:

                st.info(
                    "No new developments "
                    "were identified."
                )


            else:

                st.dataframe(
                    developments_df,
                    hide_index=True,
                    use_container_width=True
                )