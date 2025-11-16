# streamlit_app.py
import os
import io
import base64
import time
import json
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st

# ---------------------------
# Config & Secrets
# ---------------------------
st.set_page_config(page_title="Harmony Costulator + Pictator (AI OCR)", layout="wide")
st.title("⚡ Harmony Costulator + Pictator Analyzer (AI OCR)")

# Read secrets from Streamlit Secrets; allow sidebar overrides for local testing
def get_secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

OPENROUTER_KEY = get_secret("OPENROUTER_KEY", None)
HF_TOKEN = get_secret("HF_TOKEN", None)

st.sidebar.header("API Keys (optional override for local dev)")
OPENROUTER_KEY = st.sidebar.text_input("OpenRouter Key", OPENROUTER_KEY, type="password")
HF_TOKEN = st.sidebar.text_input("HuggingFace Token (HF_TOKEN)", HF_TOKEN, type="password")

if not OPENROUTER_KEY:
    st.sidebar.warning("OpenRouter key not found in secrets. OCR/analysis will not work until provided.")
if not HF_TOKEN:
    st.sidebar.info("HuggingFace token not found in secrets — Tab 4 HF image generation will fail without it.")

# ---------------------------
# OpenRouter (OpenAI-compatible) helper
# ---------------------------
try:
    import openai
    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False

def call_openrouter_model(prompt: str, model: str, api_key: str, max_tokens: int = 1500, temperature: float = 0.2):
    """
    Calls OpenRouter (OpenAI-compatible) ChatCompletion. Returns text or an error string.
    """
    if not api_key:
        return "[❌ OpenRouter API key missing]"
    if not HAS_OPENAI_SDK:
        return "[❌ OpenAI SDK missing. pip install openai==0.28.0]"

    try:
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a structured business & cost analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[❌ Model call failed: {str(e)}]"

# Fallback sequence for analysis models (tries in order, returns first non-error text)
ANALYSIS_FALLBACK_MODELS = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "mistralai/mistral-7b-instruct:free"
]

def call_openrouter_with_fallback(prompt: str, api_key: str):
    """Try multiple analysis models, return the first plausible answer."""
    for m in ANALYSIS_FALLBACK_MODELS:
        out = call_openrouter_model(prompt, m, api_key)
        if isinstance(out, str) and out.startswith("[❌"):
            # failed — try next
            continue
        # return whatever we got (may be text or csv)
        return out
    return "[❌ All analysis models failed]"

# ---------------------------
# AI OCR helpers (OpenRouter vision models)
# ---------------------------
OCR_MODELS = [
    "google/gemini-2.5-flash-lite",  # primary
    "01-ai/yi-vision"               # fallback
]

def ai_ocr_openrouter_from_image_bytes(image_bytes: bytes, api_key: str) -> str:
    """
    Send image bytes to OpenRouter vision models using ChatCompletion with
    an input_image + instruction. Tries primary then fallback.
    Returns extracted plain text or an error string.
    """
    if not api_key:
        return "[OCR unavailable — OPENROUTER_KEY not provided]"

    if not HAS_OPENAI_SDK:
        return "[OCR unavailable — OpenAI SDK missing (used for OpenRouter). Install openai package.]"

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    for model in OCR_MODELS:
        try:
            openai.api_base = "https://openrouter.ai/api/v1"
            openai.api_key = api_key
            # The following message uses a multimodal pattern: image included as base64 in user content.
            messages = [
                {"role": "system", "content": "You are an OCR assistant. Extract all visible text and table content precisely. Output plain text or CSV when table-like content is detected."},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "Extract all visible text (and tables) from the image. Output plain text; if a clean table is present, return CSV only."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"}
                ]}
            ]
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=3000
            )
            content = resp["choices"][0]["message"]["content"]
            if content and isinstance(content, str) and content.strip():
                return content
        except Exception as e:
            # try next model
            # don't expose internal error to UI, just continue
            continue

    return "[OCR FAILED — no OCR model returned usable text]"


def pdf_to_images_bytes(pdf_file_bytes: bytes, dpi: int = 200):
    """
    Convert PDF bytes to a list of PNG image bytes using PyMuPDF (fitz).
    It's robust in serverless/Cloud environments when PyMuPDF is installed.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None, "[ERROR: PyMuPDF (fitz) not installed]"

    images = []
    try:
        doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            images.append(pix.tobytes("png"))
        doc.close()
        return images, None
    except Exception as e:
        return None, f"[PDF->Image error: {e}]"

# ---------------------------
# Hugging Face Router image generation helper (Tab 4)
# ---------------------------
HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

def hf_router_generate_image(model_repo: str, prompt: str, hf_token: str, width=1024, height=1024, steps=30, guidance=3.5):
    if not hf_token:
        return {"type": "error", "data": "[HF_TOKEN missing]"}
    url = f"{HF_ROUTER_BASE}/{model_repo}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance
        }
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except Exception as e:
        return {"type": "error", "data": f"[HF Router request failed: {e}]"}
    # binary image
    ctype = resp.headers.get("content-type", "")
    if resp.status_code == 200 and "image" in ctype:
        try:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return {"type": "image", "data": img}
        except Exception as e:
            return {"type": "error", "data": f"[HF decode image failed: {e}]"}
    # try json
    try:
        data = resp.json()
    except Exception:
        return {"type": "error", "data": f"[HF non-JSON response]: {resp.text[:400]}"}
    # handle common json patterns
    try:
        if isinstance(data, dict) and "generated_image" in data:
            img_bytes = base64.b64decode(data["generated_image"])
            return {"type": "image", "data": Image.open(io.BytesIO(img_bytes)).convert("RGB")}
        if isinstance(data, dict) and "images" in data and len(data["images"])>0:
            img_bytes = base64.b64decode(data["images"][0])
            return {"type": "image", "data": Image.open(io.BytesIO(img_bytes)).convert("RGB")}
        # router may return list
        if isinstance(data, list) and len(data)>0 and isinstance(data[0], dict):
            for key in ("generated_image","blob","image"):
                if key in data[0]:
                    img_bytes = base64.b64decode(data[0][key])
                    return {"type": "image", "data": Image.open(io.BytesIO(img_bytes)).convert("RGB")}
    except Exception as e:
        return {"type": "error", "data": f"[HF parse error: {e}]"}
    return {"type": "error", "data": f"[HF unsupported response: {str(data)[:400]}]"}

# ---------------------------
# Utility: try parse model CSV output to DataFrame,
# with fallback heuristic generator if parsing fails.
# ---------------------------
def try_parse_csv_text_to_df(csv_text: str):
    """Attempt to read CSV string into DataFrame. Return (df, error)."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        return df, None
    except Exception as e:
        return None, str(e)

def heuristic_make_projections_from_df(df: pd.DataFrame):
    """
    If model fails, produce simple projections:
    - cost_0_3m = base * 0.97
    - cost_3_6m = base * 0.96
    - cost_6_9m = base * 0.95
    - cost_9_12m = base * 0.94
    Choose numeric baseline column heuristically.
    Returns new_df (with additional columns).
    """
    df2 = df.copy()
    # find baseline numeric column (prefer 'cost' or first numeric)
    baseline_col = None
    if "cost" in df2.columns:
        baseline_col = "cost"
    else:
        numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
        if numeric_cols:
            baseline_col = numeric_cols[0]
    if baseline_col is None:
        # cannot project, add zero columns
        df2["cost_0_3m"] = 0.0
        df2["cost_3_6m"] = 0.0
        df2["cost_6_9m"] = 0.0
        df2["cost_9_12m"] = 0.0
        return df2

    base = df2[baseline_col].astype(float)
    df2["cost_0_3m"] = (base * 0.97).round(2)
    df2["cost_3_6m"] = (base * 0.96).round(2)
    df2["cost_6_9m"] = (base * 0.95).round(2)
    df2["cost_9_12m"] = (base * 0.94).round(2)
    return df2

# ---------------------------
# UI Tabs
# ---------------------------
tabs = st.tabs(["🧠 Pictator Analyzer", "📊 Costulator (Profitability)", "📈 Costulator Generator", "🎨 Pictator Creator"])

# ---------------------------
# TAB 1: Pictator Analyzer
# ---------------------------
with tabs[0]:
    st.subheader("Upload Drawing / Design Image or PDF")
    st.write("AI OCR (Gemini primary, Yi-Vision fallback) will extract text. Then the analyzer will run via OpenRouter.")
    file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])
    custom_prompt = st.text_area("Custom Prompt (editable)",
                                "Analyze this engineering drawing for materials, machining process, tooling setup, optimization, and improvements.")
    enable_ai_ocr = st.checkbox("Enable AI OCR mode (no Tesseract) — uses OpenRouter vision models", value=True)

    extracted_text = ""
    if file:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            pdf_bytes = file.read()
            pages, pdf_err = pdf_to_images_bytes(pdf_bytes)
            if pdf_err:
                st.error(pdf_err)
                extracted_text = "[PDF->image conversion failed]"
            else:
                # OCR each page and concatenate
                ocr_texts = []
                with st.spinner("Running AI OCR on PDF pages..."):
                    for pbytes in pages:
                        t = ai_ocr_openrouter_from_image_bytes(pbytes, OPENROUTER_KEY) if enable_ai_ocr else "[AI OCR disabled]"
                        ocr_texts.append(t)
                extracted_text = "\n\n".join(ocr_texts)
        else:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Uploaded Drawing", use_column_width=True)
            with st.spinner("Running AI OCR..."):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                extracted_text = ai_ocr_openrouter_from_image_bytes(buf.getvalue(), OPENROUTER_KEY) if enable_ai_ocr else "[AI OCR disabled]"

        st.text_area("📜 Extracted Text", extracted_text, height=220)

        if st.button("🔍 Run Pictator Analysis"):
            st.info("Running Pictator Analyzer (with fallback models)...")
            # Build summarization & engineering analysis prompts
            summ_prompt = f"Summarize this drawing in concise engineering bullets:\n\n{extracted_text}"
            analysis_prompt = f"{custom_prompt}\n\nDrawing text:\n{extracted_text}"
            # Try summarization with fallback
            summary = call_openrouter_with_fallback(summ_prompt, OPENROUTER_KEY)
            st.markdown("### 📘 Drawing Summary")
            st.write(summary)
            # Try detailed analysis with fallback
            analysis_out = call_openrouter_with_fallback(analysis_prompt, OPENROUTER_KEY)
            st.subheader("🧩 Pictator AI Engineering Insights")
            st.write(analysis_out)
            # store for downstream if needed
            st.session_state["pictator_analyzer_text"] = extracted_text
            st.session_state["pictator_summary"] = summary
            st.session_state["pictator_analysis"] = analysis_out
            st.download_button("⬇️ Download Pictator Analysis", data=analysis_out, file_name="pictator_analysis.txt")

# ---------------------------
# TAB 2: Costulator (Profitability)
# ---------------------------
with tabs[1]:
    st.subheader("Upload Costing Sheet or Report (CSV preferred)")
    st.write("Upload the CSV exactly as the customer provided. AI OCR will extract table if image/PDF; then analysis will run and attempt to return a revised CSV (with projection columns).")
    cost_file = st.file_uploader("Upload costing image, CSV, or PDF", type=["csv", "jpg", "jpeg", "png", "pdf"])
    cost_prompt = st.text_area("Custom Prompt",
                               "Analyze this costing sheet for profitability and generate a 3–9 month cost optimization plan.")
    enable_ai_ocr2 = st.checkbox("Enable AI OCR for cost images (no Tesseract)", value=True)

    df = None
    text_data = ""
    if cost_file:
        ext = cost_file.name.split(".")[-1].lower()
        if ext == "csv":
            try:
                df = pd.read_csv(cost_file)
                st.subheader("Uploaded Table (as parsed from CSV)")
                st.dataframe(df)
                text_data = df.to_csv(index=False)
            except Exception as e:
                st.error(f"[CSV parse error: {e}]")
                df = None
                text_data = ""
        elif ext == "pdf":
            pdf_bytes = cost_file.read()
            pages, pdf_err = pdf_to_images_bytes(pdf_bytes)
            if pdf_err:
                st.error(pdf_err)
                text_data = "[PDF->image conversion failed]"
            else:
                with st.spinner("Running AI OCR on PDF pages..."):
                    page_texts = []
                    for p in pages:
                        page_texts.append(ai_ocr_openrouter_from_image_bytes(p, OPENROUTER_KEY) if enable_ai_ocr2 else "[AI OCR disabled]")
                    text_data = "\n\n".join(page_texts)
        else:
            img = Image.open(cost_file).convert("RGB")
            st.image(img, caption="Uploaded Costing Image", use_column_width=True)
            with st.spinner("Running AI OCR on image..."):
                buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
                text_data = ai_ocr_openrouter_from_image_bytes(buf.getvalue(), OPENROUTER_KEY) if enable_ai_ocr2 else "[AI OCR disabled]"
        st.text_area("🧾 Extracted Cost Data / Table (raw)", text_data, height=220)

        if st.button("💰 Run Costulator Analysis"):
            st.info("Running Costulator Analysis and generating revised CSV with projections (tries model output then fallback heuristic).")
            # Summarize cost sheet first
            summ_prompt = f"Summarize costing table and call out major cost components. Input (CSV or text):\n\n{text_data}"
            summary = call_openrouter_with_fallback(summ_prompt, OPENROUTER_KEY)
            st.markdown("### 📊 Cost Summary")
            st.write(summary)

            # Build the analysis prompt asking for a revised CSV with new columns
            analysis_prompt = f"""
You are given this cost data or description (CSV or text):
{text_data}

Produce a CSV table preserving original columns and adding these columns:
- cost_0_3m
- cost_3_6m
- cost_6_9m
- cost_9_12m

Each new column should be numeric estimates of costs after savings/optimizations.
Return only CSV text without extra commentary.
"""
            revised_csv_text = call_openrouter_with_fallback(analysis_prompt, OPENROUTER_KEY)

            # If model returns an error string (starts with [❌) treat as failure
            if isinstance(revised_csv_text, str) and revised_csv_text.startswith("[❌"):
                st.error("Analysis model failed to produce revised CSV. Falling back to heuristic projection if table available.")
                revised_df = None
            else:
                # Try to parse CSV
                revised_df, parse_err = try_parse_csv_text_to_df(revised_csv_text)
                if parse_err:
                    st.warning(f"Could not parse model CSV output: {parse_err}. Will attempt heuristic projection if original table exists.")
                    revised_df = None

            # If revised_df is None and we have an original df, create heuristic projection
            if revised_df is None and df is not None:
                revised_df = heuristic_make_projections_from_df(df)
                st.info("Used heuristic projection based on uploaded CSV (model output unavailable or unparsable).")
                revised_csv_text = revised_df.to_csv(index=False)

            if revised_df is None and df is None:
                st.error("No structured table available to project. Please upload CSV or a clear tabular image/PDF.")
            else:
                st.subheader("✅ Revised Cost Table (AI-generated / Heuristic)")
                st.dataframe(revised_df)
                st.session_state["revised_csv_text"] = revised_csv_text
                st.session_state["revised_df"] = revised_df
                st.download_button("Download Revised CSV", data=revised_csv_text, file_name="revised_costs_full.csv", mime="text/csv")

            # Store analysis outputs for Tab 3
            st.session_state["costulator_summary"] = summary
            st.session_state["costulator_analysis_text"] = revised_csv_text

# ---------------------------
# TAB 3: Costulator Generator (Auto Forecast & Excel Snapshots)
# ---------------------------
with tabs[2]:
    st.subheader("📈 Generate Forecasted Cost Sheets & Comparison")
    st.write("This reproduces the same table format you uploaded (Tab 2) and creates 6-month and 9-month optimized sheets plus a savings comparison sheet.")
    if "revised_df" not in st.session_state or st.session_state["revised_df"] is None:
        st.info("Please run Tab 2 Costulator Analysis first to generate revised cost table.")
    else:
        base_df = st.session_state["revised_df"].copy()

        # Ensure projection columns exist; if not, try to create heuristically
        required_cols = ["cost_0_3m", "cost_3_6m", "cost_6_9m", "cost_9_12m"]
        missing = [c for c in required_cols if c not in base_df.columns]
        if missing:
            base_df = heuristic_make_projections_from_df(base_df)

        # 6-month snapshot = cost_0_3m + cost_3_6m
        snapshot_6m = base_df.copy()
        snapshot_6m["cost_6m_total"] = snapshot_6m["cost_0_3m"].astype(float) + snapshot_6m["cost_3_6m"].astype(float)

        # 9-month snapshot = cost_0_3m + cost_3_6m + cost_6_9m
        snapshot_9m = base_df.copy()
        snapshot_9m["cost_9m_total"] = snapshot_9m["cost_0_3m"].astype(float) + snapshot_9m["cost_3_6m"].astype(float) + snapshot_9m["cost_6_9m"].astype(float)

        # Determine baseline (original) column for comparisons:
        baseline_col = None
        if "cost" in base_df.columns:
            baseline_col = "cost"
        else:
            numeric_cols = [c for c in base_df.columns if pd.api.types.is_numeric_dtype(base_df[c])]
            baseline_col = numeric_cols[0] if numeric_cols else None

        comp_table = pd.DataFrame()
        comp_table["item"] = base_df.index.astype(str)

        if baseline_col:
            comp_table["baseline_total"] = base_df[baseline_col].astype(float)
            comp_table["6m_total"] = snapshot_6m["cost_6m_total"]
            comp_table["9m_total"] = snapshot_9m["cost_9m_total"]
            comp_table["pct_savings_6m"] = (100.0 * (comp_table["baseline_total"] - comp_table["6m_total"]) / comp_table["baseline_total"].replace(0, np.nan)).round(2)
            comp_table["pct_savings_9m"] = (100.0 * (comp_table["baseline_total"] - comp_table["9m_total"]) / comp_table["baseline_total"].replace(0, np.nan)).round(2)
            comp_table["rs_saved_0_3"] = (base_df[baseline_col].astype(float) - base_df["cost_0_3m"].astype(float)).round(2)
            comp_table["rs_saved_3_6"] = (base_df["cost_0_3m"].astype(float) - base_df["cost_3_6m"].astype(float)).round(2)
            comp_table["rs_saved_6_9"] = (base_df["cost_3_6m"].astype(float) - base_df["cost_6_9m"].astype(float)).round(2)
        else:
            comp_table["6m_total"] = snapshot_6m["cost_6m_total"]
            comp_table["9m_total"] = snapshot_9m["cost_9m_total"]

        st.subheader("✅ Comparison Table (percentage savings)")
        st.dataframe(comp_table.fillna("N/A"))

        # Export as Excel files preserving the original format (headers & rows)
        towrite_6 = io.BytesIO()
        towrite_9 = io.BytesIO()
        with pd.ExcelWriter(towrite_6, engine="xlsxwriter") as writer:
            snapshot_6m.to_excel(writer, sheet_name="6_month_snapshot", index=False)
            comp_table.to_excel(writer, sheet_name="comparison", index=False)
        towrite_6.seek(0)

        with pd.ExcelWriter(towrite_9, engine="xlsxwriter") as writer:
            snapshot_9m.to_excel(writer, sheet_name="9_month_snapshot", index=False)
            comp_table.to_excel(writer, sheet_name="comparison", index=False)
        towrite_9.seek(0)

        st.download_button("Download 6-month Excel", data=towrite_6.getvalue(),
                           file_name="costs_6_months.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download 9-month Excel", data=towrite_9.getvalue(),
                           file_name="costs_9_months.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------
# TAB 4: Pictator Creator
# ---------------------------
with tabs[3]:
    st.subheader("Create Engineering Drawing from Text (HF Router)")
    st.write("HF models used for Pictator Creator. Uses HF_TOKEN from secrets (or sidebar override).")
    MODELS = {
        "Sketchers (Lineart / Mechanical)": "black-forest-labs/FLUX.1-dev",
        "CAD Drawing XL (2D CNC Blueprints)": "stabilityai/stable-diffusion-xl-base-1.0",
        "RealisticVision (3D Render)": "stabilityai/stable-diffusion-3-medium-diffusers"
    }
    model_choice = st.selectbox("Choose HF model (Pictator)", list(MODELS.keys()))
    prompt = st.text_area("Drawing Prompt", "technical CNC blueprint lineart of disc brake, top view, thin black lines, engineering drawing")
    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input("Width", min_value=256, max_value=1536, value=768)
    with col2:
        height = st.number_input("Height", min_value=256, max_value=1536, value=768)
    steps = st.slider("Inference Steps", 5, 80, 30)
    guidance = st.slider("Guidance Scale", 1.0, 12.0, 3.5)

    if st.button("Generate Pictator Image"):
        with st.spinner("Generating image on Hugging Face Router..."):
            repo = MODELS[model_choice]
            out = hf_router_generate_image(repo, prompt, HF_TOKEN, width=width, height=height, steps=steps, guidance=guidance)
        if out.get("type") == "image":
            img = out["data"]
            st.image(img, caption="Generated Pictator Drawing", use_column_width=True)
            buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
            st.download_button("Download PNG", data=buf.getvalue(), file_name="pictator.png", mime="image/png")
        else:
            st.error(out.get("data", "Unknown HF Router error"))

st.markdown("---")
st.caption("© 2025 Harmony Strategy Partner — Costulator + Pictator Suite (AI OCR via OpenRouter)")
