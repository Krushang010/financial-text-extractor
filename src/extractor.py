import re
import time
import pandas as pd

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=2000
)


# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------

prompt = """
You are a financial information extraction assistant.

The text below may be ONE CHUNK from a larger earnings transcript,
financial article, conference call, or company update.

Extract only information that is explicitly supported by this chunk.

FINANCIAL INFORMATION

Extract:

1. company
2. reporting_period
3. revenue_actual
4. revenue_expected
5. revenue_growth
6. eps_actual
7. eps_expected
8. profit_growth
9. margin
10. margin_change
11. capex
12. capacity_expansion
13. management_guidance
14. key_positive
15. key_risk


WHAT IS NEW

Identify important NEW developments mentioned in this chunk.

Examples include:

- Acquisition
- Merger
- Management Change
- New CEO / CFO / Director
- New Product
- Product Launch
- New Order
- Contract Win
- New Client
- Partnership
- Joint Venture
- Capex
- Capacity Expansion
- New Plant
- New Facility
- New Market
- New Geography
- New Business Segment
- Investment
- Fund Raising
- Technology Development
- Subsidiary Formation
- Strategic Initiative
- Other important new development


Return new developments under:

"new_developments"

Every development must contain:

{{
    "category": "...",
    "detail": "..."
}}


RULES

1. Never invent information.
2. Extract only facts supported by the supplied text.
3. If a financial field is not mentioned, return "Not Mentioned".
4. If there are no new developments, return an empty list.
5. Keep original financial units such as crore, million, billion and %.
6. Preserve YoY, QoQ or other growth descriptions when available.
7. Do not treat ordinary historical information as a new development.
8. A development should genuinely indicate something new, announced,
   launched, appointed, acquired, secured, planned, commissioned,
   expanded, entered, signed or introduced.
9. Historical comparison numbers should not replace the current-period
   number when the current-period number is available.
10. Do not infer information requiring another part of the transcript.
11. Return valid JSON only.
12. Do not return markdown, commentary or explanation.

Your JSON should follow this structure:

{{
    "company": "...",
    "reporting_period": "...",
    "revenue_actual": "...",
    "revenue_expected": "...",
    "revenue_growth": "...",
    "eps_actual": "...",
    "eps_expected": "...",
    "profit_growth": "...",
    "margin": "...",
    "margin_change": "...",
    "capex": "...",
    "capacity_expansion": "...",
    "management_guidance": "...",
    "key_positive": "...",
    "key_risk": "...",
    "new_developments": [
        {{
            "category": "...",
            "detail": "..."
        }}
    ]
}}

Financial Text:
{text}
"""


prompt_template = PromptTemplate.from_template(prompt)

parser = JsonOutputParser()

chain = prompt_template | llm | parser


# ---------------------------------------------------------
# TEXT SPLITTER
# ---------------------------------------------------------

# We deliberately keep chunks fairly small because the free
# Groq tier has a tokens-per-minute limit.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=300,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# ---------------------------------------------------------
# FIELD GROUPS
# ---------------------------------------------------------

SINGLE_VALUE_FIELDS = [
    "company",
    "reporting_period",
    "revenue_actual",
    "revenue_expected",
    "revenue_growth",
    "eps_actual",
    "eps_expected",
    "profit_growth",
    "margin",
    "margin_change"
]


MULTI_VALUE_FIELDS = [
    "capex",
    "capacity_expansion",
    "management_guidance",
    "key_positive",
    "key_risk"
]


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def is_missing(value):
    """
    Check whether an extracted field contains a meaningful value.
    """

    if value is None:
        return True

    value = str(value).strip()

    return value.lower() in {
        "",
        "not mentioned",
        "not available",
        "n/a",
        "none",
        "null"
    }


def count_congratulations(text):
    """
    Count the exact word 'congratulations' without using the LLM.
    """

    return len(
        re.findall(
            r"\bcongratulations\b",
            text,
            flags=re.IGNORECASE
        )
    )


# ---------------------------------------------------------
# MERGE CHUNK RESULTS
# ---------------------------------------------------------

def merge_chunk_results(results):

    merged = {
        field: "Not Mentioned"
        for field in SINGLE_VALUE_FIELDS
    }

    collected_values = {
        field: []
        for field in MULTI_VALUE_FIELDS
    }

    developments = []

    seen_developments = set()


    for result in results:

        if not isinstance(result, dict):
            continue


        # -------------------------------------------------
        # SINGLE VALUE FIELDS
        # -------------------------------------------------

        for field in SINGLE_VALUE_FIELDS:

            value = result.get(field)

            if (
                merged[field] == "Not Mentioned"
                and not is_missing(value)
            ):

                merged[field] = str(value).strip()


        # -------------------------------------------------
        # MULTI VALUE FIELDS
        # -------------------------------------------------

        for field in MULTI_VALUE_FIELDS:

            value = result.get(field)

            if is_missing(value):
                continue

            value = str(value).strip()

            if value not in collected_values[field]:

                collected_values[field].append(value)


        # -------------------------------------------------
        # NEW DEVELOPMENTS
        # -------------------------------------------------

        chunk_developments = result.get(
            "new_developments",
            []
        )

        if not isinstance(chunk_developments, list):
            continue


        for item in chunk_developments:

            if not isinstance(item, dict):
                continue


            category = str(
                item.get(
                    "category",
                    "Other"
                )
            ).strip()


            detail = str(
                item.get(
                    "detail",
                    ""
                )
            ).strip()


            if not detail:
                continue


            # Used for simple duplicate removal

            duplicate_key = (
                category.lower(),
                detail.lower()
            )


            if duplicate_key not in seen_developments:

                seen_developments.add(
                    duplicate_key
                )

                developments.append(
                    {
                        "category": category,
                        "detail": detail
                    }
                )


    # -----------------------------------------------------
    # COMBINE MULTI-VALUE FIELDS
    # -----------------------------------------------------

    for field, values in collected_values.items():

        if values:

            merged[field] = " | ".join(values)

        else:

            merged[field] = "Not Mentioned"


    merged["new_developments"] = developments

    return merged


# ---------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ---------------------------------------------------------

def extract_financial_data(
    text,
    progress_callback=None
):

    if not text or not text.strip():

        return {
            "error": "No text was provided."
        }


    # -----------------------------------------------------
    # DETERMINISTIC PYTHON COUNT
    # -----------------------------------------------------

    congratulations_count = (
        count_congratulations(text)
    )


    # -----------------------------------------------------
    # SPLIT LONG TRANSCRIPT
    # -----------------------------------------------------

    chunks = text_splitter.split_text(text)

    chunk_results = []

    skipped_chunks = 0


    try:

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            # -------------------------------------------------
            # GROQ FREE-TIER THROTTLING
            #
            # Multiple smaller requests can still collectively
            # exceed the TPM limit.
            # -------------------------------------------------

            if index > 1:

                time.sleep(22)


            try:

                chunk_result = chain.invoke(
                    {
                        "text": chunk
                    }
                )

                chunk_results.append(
                    chunk_result
                )



            except OutputParserException as e:

                skipped_chunks += 1

                print(

                    f"\nChunk {index} failed JSON parsing:"

                )

                print(e)


            # -------------------------------------------------
            # STREAMLIT PROGRESS CALLBACK
            # -------------------------------------------------

            if progress_callback:

                progress_callback(
                    index,
                    len(chunks)
                )


        # -----------------------------------------------------
        # MERGE ALL CHUNK EXTRACTIONS
        # -----------------------------------------------------

        final_result = merge_chunk_results(
            chunk_results
        )


        final_result[
            "congratulations_count"
        ] = congratulations_count


        final_result[
            "chunks_processed"
        ] = len(chunks)


        final_result[
            "chunks_skipped"
        ] = skipped_chunks


        return final_result


    except Exception as e:

        error_message = str(e)


        # More useful message for Groq rate limits

        if (
            "rate_limit" in error_message.lower()
            or "429" in error_message
            or "413" in error_message
        ):

            return {
                "error":
                    "Groq free-tier rate limit was reached.",
                "details":
                    "Wait around one minute and try again. "
                    "The transcript is already being chunked, "
                    "but the free API also limits total tokens "
                    "processed per minute."
            }


        return {
            "error":
                "Something went wrong while processing the text.",

            "details":
                error_message
        }


# ---------------------------------------------------------
# FINANCIAL TABLE
# ---------------------------------------------------------

def financial_to_dataframe(data):

    fields = [
        "company",
        "reporting_period",
        "revenue_actual",
        "revenue_expected",
        "revenue_growth",
        "eps_actual",
        "eps_expected",
        "profit_growth",
        "margin",
        "margin_change",
        "capex",
        "capacity_expansion",
        "management_guidance",
        "key_positive",
        "key_risk"
    ]


    rows = []


    for field in fields:

        rows.append(
            {
                "Field":
                    field
                    .replace("_", " ")
                    .title(),

                "Value":
                    data.get(
                        field,
                        "Not Mentioned"
                    )
            }
        )


    return pd.DataFrame(rows)


# ---------------------------------------------------------
# DEVELOPMENTS TABLE
# ---------------------------------------------------------

def developments_to_dataframe(data):

    developments = data.get(
        "new_developments",
        []
    )


    if not developments:

        return pd.DataFrame(
            columns=[
                "Category",
                "Development"
            ]
        )


    rows = []


    for item in developments:

        if not isinstance(item, dict):
            continue


        rows.append(
            {
                "Category":
                    item.get(
                        "category",
                        "Other"
                    ),

                "Development":
                    item.get(
                        "detail",
                        "Not Mentioned"
                    )
            }
        )


    return pd.DataFrame(rows)