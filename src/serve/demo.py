"""Streamlit demo application for EdgarLENS financial extraction."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("EDGARLENS_API_URL", "http://localhost:8000")

SAMPLE_CHUNK = """ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
Apple Inc.
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except number of shares which are reflected in thousands and per share amounts)

Years ended September 28, 2024
Total net sales: $ 391,035
Cost of sales: 210,352
Gross margin: 180,683
Operating expenses:
  Research and development: 31,370
  Selling, general and administrative: 26,050
Total operating expenses: 57,420
Operating income: 123,263
Other income/(expense), net: 269
Income before provision for income taxes: 123,532
Provision for income taxes: 29,796
Net income: $ 93,736

Earnings per share:
  Basic: $ 6.11
  Diluted: $ 6.08
Shares used in computing earnings per share:
  Diluted: 15,408,000
"""

st.set_page_config(
    page_title="EdgarLENS | Financial Extractor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 EdgarLENS Financial Extraction Gateway")
st.markdown(
    "**Fine-tuned 7B LLM with schema-guided decoding, post-validation financial sanity checks, "
    "and automated frontier model fallback.**"
)

# Sidebar settings
st.sidebar.header("Gateway Configuration")
gateway_url = st.sidebar.text_input("FastAPI Gateway URL", value=API_URL)
allow_fallback = st.sidebar.checkbox("Allow Frontier Fallback on Failure", value=True)
strict_validation = st.sidebar.checkbox("Enforce Strict Sanity Checks", value=True)

# Check Health
try:
    resp = httpx.get(f"{gateway_url}/health", timeout=3.0)
    if resp.status_code == 200:
        health_info = resp.json()
        st.sidebar.success(f"Gateway Online: {health_info.get('status', 'unknown')}")
        st.sidebar.caption(
            f"Model: `{health_info.get('model_name')}` | Fallback Ready: `{health_info.get('fallback_available')}`"
        )
    else:
        st.sidebar.warning(f"Gateway returned {resp.status_code}")
except Exception:
    st.sidebar.error("Gateway Offline (Start API server on port 8000)")

tab_text, tab_accn = st.tabs(["📄 Paste Filing Text / Chunk", "🔍 SEC Accession Number"])

payload = None

with tab_text:
    text_input = st.text_area("Filing Text / Markdown Table", value=SAMPLE_CHUNK, height=260)
    if st.button("Extract from Text", type="primary") and text_input.strip():
        payload = {
            "text": text_input.strip(),
            "allow_fallback": allow_fallback,
            "strict_validation": strict_validation,
        }

with tab_accn:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        accn_input = st.text_input("SEC Accession Number", value="0000320193-24-000106")
    with col_b:
        ticker_input = st.text_input("Ticker", value="AAPL")

    if st.button("Lookup & Extract Filing", type="primary") and accn_input.strip():
        payload = {
            "accession_number": accn_input.strip(),
            "ticker": ticker_input.strip() if ticker_input.strip() else None,
            "allow_fallback": allow_fallback,
            "strict_validation": strict_validation,
        }

if payload:
    with st.spinner("Extracting financial metrics and verifying financial sanity..."):
        try:
            res = httpx.post(f"{gateway_url}/extract", json=payload, timeout=60.0)
            if res.status_code == 200:
                data = res.json()
                st.success("Extraction Completed!")

                # Key Metrics Bar
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latency", f"{data['latency_ms']:.1f} ms")
                m2.metric("Model Used", data["model"])
                val_status = "✓ Passed" if data["validation"]["passed"] else "⚠ Failed"
                m3.metric("Sanity Checks", val_status)
                fb_status = "Yes (Frontier)" if data["fallback_used"] else "No (Primary vLLM)"
                m4.metric("Fallback Triggered", fb_status)

                # Validation Details
                if not data["validation"]["passed"]:
                    st.error("Validation Violations:")
                    for err in data["validation"]["errors"]:
                        st.markdown(f"- ❌ {err}")

                if data["validation"].get("warnings"):
                    st.warning("Validation Warnings:")
                    for w in data["validation"]["warnings"]:
                        st.markdown(f"- ⚠️ {w}")

                # Display Concepts Table
                metrics_dict = data["metrics"]
                table_rows = []
                for concept, metric in metrics_dict.items():
                    if metric:
                        val = metric.get("value")
                        formatted_val = (
                            f"${val:,.2f}" if concept != "diluted_eps" else f"${val:.2f}"
                        )
                        period = metric.get("period_type", "-")
                        table_rows.append(
                            {"Concept": concept, "Extracted Value": formatted_val, "Period": period}
                        )
                    else:
                        table_rows.append(
                            {
                                "Concept": concept,
                                "Extracted Value": "null (not in text)",
                                "Period": "-",
                            }
                        )

                st.subheader("Extracted Financial Metrics")
                st.dataframe(table_rows, use_container_width=True)

                with st.expander("Raw JSON Response & Tracing"):
                    st.json(data)
            else:
                st.error(f"API Error ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
