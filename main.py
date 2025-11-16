# streamlit_app.py
# Harmony Costulator + Pictator Analyzer — Streamlit Cloud edition
# Requires: OPENROUTER_KEY and HF_TOKEN set in Streamlit Secrets

import os
import io
import re
import json
import time
import base64
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st

# Optional local tesseract - won't run on Streamlit Cloud unless installed in environment
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

# ----------------------
# Secrets (Streamlit Cloud)
# ----------------------
try:
    OPENROUTER_KEY = st.secrets["OPENROUTER_KEY"]
except Exception:
    OPENROUTER_KEY = None

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = None

# Sidebar quick info
st.sidebar.header("API Keys")
st.sidebar.write("OpenRouter key present:", bool(OPENROUTER_KEY))
st.sidebar.write("HF token present:", bool(HF_TOKEN))

# ----------------------
# OpenRouter (OpenAI-compatible) wrapper using openai SDK if available
# ----------------------
HAS_OPENAI_SDK = False
try:
    import openai
    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False

# analysis fallback models (try in order)
ANALYZER_MODELS = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-7b-instruct:free"
]

# OCR model order (OpenRouter-hosted names)
OCR_MODELS = [
    "google/gemini-2.5-flash-lite",
    "01-ai/yi-vision"
]

def call_openrouter_chat(prompt: str, model: str, api_key: str, max_tokens: int = 1500, temperature: float = 0.2):
    """Call OpenRouter-compatible ChatCompletion via openai SDK. Returns text or error string."""
    if not api_key:
        return "[❌ OpenRouter API key missing]"
    if not HAS_OPENAI_SDK:
        return "[❌ OpenAI SDK missing: pip install openai==0.28.0]"
    try:
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a structured business & cost analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[❌ Model call failed: {str(e)}]"

def try_multiple_analyzers(prompt: str, api_key: str, models_list=ANALYZER_MODELS):
    """Try analyzer models in order until one returns a non-error response."""
    last_err = None
    for m in models_list:
        res = call_openrouter_chat(prompt, m, api_key, max_tokens=1800, temperature=0.2)
        # treat outputs starting with '[❌' as errors
        if isinstance(res, str) and not res.strip().startswith("[❌"):
            return res, m
        last_err = (m, res)
    # all failed
    return f"[❌ All analyzer models failed. Last: {last_err}]", None

# ----------------------
# OCR via OpenRouter (vision-capable models)
# We'll send image as base64 in the prompt and ask for extracted text ONLY.
# ----------------------
def ocr_via_openrouter_base64(b64_image: str, api_key: str, model_name: str):
    """Ask a vision model on OpenRouter to extract text from a base64 image. Returns extracted text or error."""
    user_msg = (
        "You are an OCR assistant. Extract all readable text from the image provided below and return plain text only.\n"
        "The image is provided as a base64 PNG data URI (data:image/png;base64,....). Do not add commentary.\n\n"
        f"IMAGE_DATA_URI:data:image/png;base64,{b64_image}"
    )
    return call_openrouter_chat(user_msg, model_name, api_key, max_tokens=2500, temperature=0.0)

def ai_ocr_image(img: Image.Image, api_key: str):
    """Try OCR models in OCR_MODELS order via OpenRouter; fallback to pytesseract if available."""
    # convert to PNG bytes + base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    for m in OCR_MODELS:
        try:
            txt = ocr_via_openrouter_base64(b64, api_key, m)
            if txt and isinstance(txt, str) and not txt.startswith("[❌"):
                return txt, m
        except Exception:
            continue
    # pytesseract fallback
    if HAS_TESSERACT:
        try:
            txt = pytesseract.image_to_string(img)
            return txt.strip() or "[No text found by Tesseract]", "tesseract"
        except Exception:
            pass
    return "[OCR failed: no model returned text]", None

# ----------------------
# Utilities: parse CSV returned by model robustly
# ----------------------
def extract_csv_from_text(text: str):
    """
    Some models wrap CSV in markdown/code fences or have commentary.
    Attempt to extract the CSV substring - first code block, then any lines with commas.
    """
    if not isinstance(text, str):
        return None
    # strip markdown fences
    code_block = re.search(r"```(?:csv|text)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        csv_text = code_block.group(1).strip()
        return csv_text
    # look for first line containing commas and multiple lines following
    lines = text.splitlines()
    # find start index where a line has comma and not just a single value
    for i, L in enumerate(lines):
        if "," in L and len(L.split(",")) > 1:
            # gather until blank line or end
            j = i
            out_lines = []
            while j < len(lines) and lines[j].strip() != "":
                out_lines.append(lines[j])
                j += 1
            candidate = "\n".join(out_lines).strip()
            # sanity: at least 2 lines and 1 comma
            if len(out_lines) >= 2:
                return candidate
    # fallback: if text itself looks like CSV
    if "," in text and "\n" in text:
        return text.strip()
    return None

# ----------------------
# Analysis prompt templates
# ----------------------
ANALYSIS_BRIEF_PROMPT = (
    "You are a cost analysis assistant. Given the following cost table (CSV) or description, "
    "summarize key cost components, list optimization steps and estimated percentage savings per component. "
    "Return plain text analysis (human readable)."
)

REVISED_CSV_PROMPT_TEMPLATE = (
    "You are a helpful assistant. I will provide a cost CSV. Produce a revised CSV that preserves ALL original columns "
    "and rows (same order) and add these numeric columns: cost_0_3m, cost_3_6m, cost_6_9m, cost_9_12m. "
    "For each row, estimate the optimized cost numbers after applying improvements (numbers only). "
    "Assume the original numeric cost column is named 'cost' if present; otherwise infer the primary cost numeric column. "
    "Return ONLY the CSV text (no commentary). Input CSV below:\n\n{csv_text}"
)

# ----------------------
# Tab UI & logic
# ----------------------
st.set_page_config(page_title="Harmony Costulator + Pictator Analyzer", layout="wide")
st.title("⚡ Harmony Costulator + Pictator Analyzer (Streamlit Cloud)")

tabs = st.tabs(["🧠 Pictator Analyzer", "📊 Costulator (Profitability)", "📈 Costulator Generator", "🎨 Pictator Creator"])

# ----------------------
# Tab 1: Pictator Analyzer (image upload and analysis - keep identical to requested behavior)
# ----------------------
with tabs[0]:
    st.header("🧠 Pictator Analyzer")
    st.write("Upload a design/drawing (image or PDF). The analyzer will OCR and summarize the drawing.")
    upload = st.file_uploader("Upload drawing (jpg/png/pdf)", type=["jpg", "jpeg", "png", "pdf"])
    custom_prompt = st.text_area("Custom prompt for analysis (optional)", "Analyze this engineering drawing for materials, machining process, tooling setup, optimization suggestions.")
    enable_ai_ocr = st.checkbox("Enable AI OCR mode (no Tesseract)", value=True)

    if upload:
        ext = upload.name.split(".")[-1].lower()
        extracted_text = ""
        ocr_model_used = None
        if ext == "pdf":
            # try simple PDF -> image conversion if pdf2image is available; otherwise read text via pdfplumber if installed
            try:
                from pdf2image import convert_from_bytes
                imgs = convert_from_bytes(upload.read())
                st.image(imgs[0], caption="First page")
                if enable_ai_ocr and OPENROUTER_KEY:
                    extracted_text, ocr_model_used = ai_ocr_image(imgs[0], OPENROUTER_KEY)
                else:
                    extracted_text = "[PDF uploaded; OCR not enabled]"
            except Exception:
                extracted_text = "[PDF processing not available in this environment]"
        else:
            img = Image.open(upload).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)
            if enable_ai_ocr and OPENROUTER_KEY:
                extracted_text, ocr_model_used = ai_ocr_image(img, OPENROUTER_KEY)
            else:
                # fallback to PIL preprocess + pytesseract
                if HAS_TESSERACT:
                    extracted_text = pytesseract.image_to_string(img)
                else:
                    extracted_text = "[OCR not enabled and Tesseract not available]"

        st.subheader("📜 Extracted Text (OCR output)")
        st.text_area("OCR output", extracted_text, height=200)
        if ocr_model_used:
            st.caption(f"OCR model used: {ocr_model_used}")

        if st.button("🔍 Run Pictator Analysis (on OCR text)"):
            if not OPENROUTER_KEY:
                st.error("OpenRouter key missing (st.secrets['OPENROUTER_KEY']). Cannot run analysis.")
            else:
                prompt = f"{ANALYSIS_BRIEF_PROMPT}\n\nOCR_TEXT:\n{extracted_text}\n\nCUSTOM_PROMPT:\n{custom_prompt}"
                analysis, model_used = try_multiple_analyzers(prompt, OPENROUTER_KEY)
                st.subheader("🧾 Pictator Analysis Summary")
                st.write(analysis)
                st.download_button("⬇️ Download Analysis (TXT)", data=analysis, file_name="pictator_analysis.txt", mime="text/plain")

# ----------------------
# Tab 2: Costulator (Profitability) — upload costing and run analysis
# ----------------------
with tabs[1]:
    st.header("📊 Costulator (Profitability)")
    st.write("Upload your cost sheet (CSV preferred). The app will render it and produce an AI analysis and revised CSV (with projections).")
    cost_file = st.file_uploader("Upload costing (CSV / image / PDF)", type=["csv", "xls", "xlsx", "jpg", "jpeg", "png", "pdf"])
    cost_custom_prompt = st.text_area("Custom prompt for cost analysis (optional)", "Analyze this costing sheet for profitability and recommend optimized costs and savings for 0-3, 3-6, 6-9 months.")
    enable_ai_ocr_cost = st.checkbox("Enable AI OCR for cost images (no Tesseract)", value=True)

    st.write("Tip: Upload a CSV to get exact table reproduction across tabs.")

    if cost_file:
        ext = cost_file.name.split(".")[-1].lower()
        df = None
        original_csv_text = None

        if ext in ("csv", "xls", "xlsx"):
            try:
                # try CSV first; if excel, use pandas read_excel
                if ext == "csv":
                    df = pd.read_csv(cost_file)
                    original_csv_text = df.to_csv(index=False)
                else:
                    df = pd.read_excel(cost_file)
                    original_csv_text = df.to_csv(index=False)
                st.subheader("✅ Uploaded table (rendered exactly)")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Failed to parse uploaded spreadsheet: {e}")
                st.stop()
        elif ext == "pdf":
            # try pdf OCR to produce table text
            try:
                from pdf2image import convert_from_bytes
                pages = convert_from_bytes(cost_file.read())
                st.image(pages[0], caption="First PDF page")
                if enable_ai_ocr_cost and OPENROUTER_KEY:
                    ocr_text, _ = ai_ocr_image(pages[0], OPENROUTER_KEY)
                    original_csv_text = ocr_text
                    st.text_area("Extracted text from PDF (preview)", ocr_text[:3000], height=200)
                else:
                    st.error("PDF parsing requires pdf2image or AI OCR enabled.")
            except Exception as e:
                st.error(f"PDF processing error: {e}")
        else:
            # image
            img = Image.open(cost_file).convert("RGB")
            st.image(img, caption="Uploaded cost image")
            if enable_ai_ocr_cost and OPENROUTER_KEY:
                ocr_text, model_used = ai_ocr_image(img, OPENROUTER_KEY)
                original_csv_text = ocr_text
                st.text_area("OCR extracted text (preview)", ocr_text[:3000], height=200)
            else:
                if HAS_TESSERACT:
                    txt = pytesseract.image_to_string(img)
                    original_csv_text = txt
                    st.text_area("Tesseract OCR text", txt[:3000], height=200)
                else:
                    st.error("OCR disabled and Tesseract not available. Provide a CSV for best results.")

        if st.button("💰 Run Costulator Analysis and Produce Revised CSV"):
            if not OPENROUTER_KEY:
                st.error("OpenRouter key missing (st.secrets['OPENROUTER_KEY']). Cannot run analysis.")
            else:
                # produce analysis summary (human readable) and generate revised CSV (CSV-only)
                # 1) Analysis summary
                analysis_prompt = (
                    f"{ANALYSIS_BRIEF_PROMPT}\n\nCost data (CSV or text):\n{original_csv_text}\n\nCustom prompt:\n{cost_custom_prompt}"
                )
                st.info("Running human-readable analysis (may try multiple models)...")
                human_analysis, model_used = try_multiple_analyzers(analysis_prompt, OPENROUTER_KEY)
                st.subheader("🔎 Cost Summary (human-readable)")
                st.write(human_analysis)
                st.caption(f"Analyzer model used: {model_used}")

                # 2) Generate revised CSV with projections
                st.info("Requesting revised CSV with projection columns (CSV-only output).")
                revised_prompt = REVISED_CSV_PROMPT_TEMPLATE.format(csv_text=original_csv_text)
                revised_csv_text, model_used_csv = try_multiple_analyzers(revised_prompt, OPENROUTER_KEY)
                st.caption(f"CSV-generator model used: {model_used_csv}")

                if revised_csv_text.startswith("[❌"):
                    st.error("Revised CSV generation failed: " + revised_csv_text)
                else:
                    # extract CSV substring from response robustly
                    csv_candidate = extract_csv_from_text(revised_csv_text)
                    if not csv_candidate:
                        csv_candidate = revised_csv_text  # try entire text

                    # try parse candidate CSV
                    try:
                        new_df = pd.read_csv(io.StringIO(csv_candidate))
                        # preserve original columns order plus new projection columns placed at end
                        st.success("Revised CSV parsed successfully.")
                        st.subheader("✅ Revised Cost Table (AI-generated projections)")
                        st.dataframe(new_df)
                        # store in session for Tab 3
                        st.session_state["revised_csv_text"] = csv_candidate
                        st.session_state["revised_df"] = new_df
                        # also store original df and its CSV (for format matching)
                        st.session_state["original_df"] = df
                        st.session_state["original_csv_text"] = original_csv_text
                        # download link
                        st.download_button("Download Revised CSV", data=csv_candidate, file_name="revised_costs_full.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Could not parse revised CSV. Error: {e}")
                        st.code(revised_csv_text[:4000])

# ----------------------
# Tab 3: Costulator Generator — exact format tables for 6-month, 9-month, and savings
# ----------------------
with tabs[2]:
    st.header("📈 Costulator Generator — 6/9 Month Snapshots & Savings")
    st.write("This reproduces the exact uploaded table format then generates 6-month and 9-month optimized tables and a savings sheet.")

    if "revised_df" not in st.session_state or st.session_state["revised_df"] is None:
        st.info("Please run Tab 2 and produce a revised CSV first (press 'Run Costulator Analysis' in Tab 2).")
    else:
        base_df = st.session_state["revised_df"].copy()
        original_df = st.session_state.get("original_df", None)

        st.subheader("Original / Revised base table (reproduced):")
        # If original_df exists, show it; else show revised
        if original_df is not None:
            st.write("Original uploaded table (as displayed in Tab 2):")
            st.dataframe(original_df)
        st.write("AI Revised table (with projection columns):")
        st.dataframe(base_df)

        # Required projection columns
        required_cols = ["cost_0_3m", "cost_3_6m", "cost_6_9m", "cost_9_12m"]
        missing = [c for c in required_cols if c not in base_df.columns]
        if missing:
            st.error(f"Revised CSV missing required projection columns: {missing}. Tab 2 must produce them.")
        else:
            # create snapshots while keeping same columns & rows. We'll add a column indicating snapshot totals.
            snap6 = base_df.copy()
            snap6["cost_6m_total"] = snap6["cost_0_3m"].astype(float) + snap6["cost_3_6m"].astype(float)

            snap9 = base_df.copy()
            snap9["cost_9m_total"] = (
                snap9["cost_0_3m"].astype(float) + snap9["cost_3_6m"].astype(float) + snap9["cost_6_9m"].astype(float)
            )

            # Determine baseline column to compute savings: prefer 'cost' if present, else first numeric column
            baseline_col = None
            if "cost" in base_df.columns:
                baseline_col = "cost"
            else:
                numeric_cols = [c for c in base_df.columns if pd.api.types.is_numeric_dtype(base_df[c])]
                if numeric_cols:
                    baseline_col = numeric_cols[0]

            comp_table = None
            if baseline_col:
                comp = pd.DataFrame()
                comp["row_index"] = base_df.index.astype(str)
                comp["baseline_total"] = base_df[baseline_col].astype(float)
                comp["6m_total"] = snap6["cost_6m_total"]
                comp["9m_total"] = snap9["cost_9m_total"]
                comp["rs_saved_0_3"] = base_df[baseline_col].astype(float) - base_df["cost_0_3m"].astype(float)
                comp["rs_saved_3_6"] = base_df["cost_3_6m"].astype(float) - base_df["cost_3_6m"].astype(float)  # typically 0 as projections are direct; leave for formula
                # For percent savings vs baseline total for 6m and 9m
                comp["pct_savings_6m"] = 100.0 * (comp["baseline_total"] - comp["6m_total"]) / comp["baseline_total"].replace(0, np.nan)
                comp["pct_savings_9m"] = 100.0 * (comp["baseline_total"] - comp["9m_total"]) / comp["baseline_total"].replace(0, np.nan)
                comp_table = comp
            else:
                # fallback simple totals
                comp_table = pd.DataFrame({
                    "row_index": base_df.index.astype(str),
                    "6m_total": snap6["cost_6m_total"],
                    "9m_total": snap9["cost_9m_total"]
                })

            st.subheader("📊 Comparison Table (savings and %)")
            st.dataframe(comp_table.fillna("N/A"))

            # Export Excel files that keep the same columns/headers and add snapshots/comparisons as separate sheets
            towrite_6 = io.BytesIO()
            towrite_9 = io.BytesIO()
            with pd.ExcelWriter(towrite_6, engine="xlsxwriter") as writer:
                # keep same column order as base_df
                base_df.to_excel(writer, sheet_name="revised_base", index=False)
                snap6.to_excel(writer, sheet_name="6_month_snapshot", index=False)
                comp_table.to_excel(writer, sheet_name="comparison", index=False)
            with pd.ExcelWriter(towrite_9, engine="xlsxwriter") as writer:
                base_df.to_excel(writer, sheet_name="revised_base", index=False)
                snap9.to_excel(writer, sheet_name="9_month_snapshot", index=False)
                comp_table.to_excel(writer, sheet_name="comparison", index=False)

            towrite_6.seek(0)
            towrite_9.seek(0)

            st.download_button("Download 6-month Excel (same format)", data=towrite_6.getvalue(),
                               file_name="costs_6_months.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("Download 9-month Excel (same format)", data=towrite_9.getvalue(),
                               file_name="costs_9_months.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------------
# Tab 4: Pictator Creator — HF Router image generation (kept as requested)
# ----------------------
with tabs[3]:
    st.header("🎨 Pictator Creator (Text → Drawing)")
    st.write("Generates engineering drawings via HuggingFace Router models (HF_TOKEN from secrets).")
    # available HF models (as requested earlier)
    HF_MODELS = {
        "Sketchers (Lineart / Mechanical)": "black-forest-labs/FLUX.1-dev",
        "CAD Drawing XL (2D CNC Blueprints)": "stabilityai/stable-diffusion-xl-base-1.0",
        "RealisticVision (3D Render)": "stabilityai/stable-diffusion-3-medium-diffusers"
    }
    pict_model_choice = st.selectbox("Choose HF model", list(HF_MODELS.keys()))
    pict_model_repo = HF_MODELS[pict_model_choice]
    pict_prompt = st.text_area("Drawing prompt", "technical blueprint lineart of a disc brake, top view, thin lines, engineering drawing")
    col1, col2 = st.columns(2)
    width = col1.number_input("Width", 256, 1536, 768)
    height = col2.number_input("Height", 256, 1536, 768)
    steps = st.slider("Inference steps", 5, 80, 30)
    guidance = st.slider("Guidance scale", 1.0, 12.0, 3.5)

    def hf_router_generate_image(hf_token: str, model_repo: str, prompt: str, width=768, height=768, steps=30, guidance=3.5):
        if not hf_token:
            return {"type": "error", "data": "[HF_TOKEN missing in st.secrets]"}
        url = f"https://router.huggingface.co/hf-inference/models/{model_repo}"
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
            r = requests.post(url, headers=headers, json=payload, timeout=120)
        except Exception as e:
            return {"type": "error", "data": f"HF Router request failed: {e}"}
        # binary image
        if "image" in r.headers.get("content-type", ""):
            try:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                return {"type": "image", "data": img}
            except Exception as e:
                return {"type": "error", "data": f"Image decode failed: {e}"}
        # json
        try:
            data = r.json()
            csv = None
            # many formats: images, generated_image, blob
            if isinstance(data, dict) and "generated_image" in data:
                b = base64.b64decode(data["generated_image"])
                return {"type": "image", "data": Image.open(io.BytesIO(b)).convert("RGB")}
            if isinstance(data, dict) and "images" in data and data["images"]:
                b = base64.b64decode(data["images"][0])
                return {"type": "image", "data": Image.open(io.BytesIO(b)).convert("RGB")}
            if isinstance(data, list) and data and isinstance(data[0], dict):
                for k in ("blob","generated_image","image"):
                    if k in data[0]:
                        b = base64.b64decode(data[0][k])
                        return {"type": "image", "data": Image.open(io.BytesIO(b)).convert("RGB")}
            return {"type": "error", "data": f"Unsupported HF Router response: {str(data)[:300]}"}
        except Exception as e:
            return {"type": "error", "data": f"HF Router non-json response or decode failed: {str(e)}"}

    if st.button("Generate Pictator Drawing"):
        out = hf_router_generate_image(HF_TOKEN, pict_model_repo, pict_prompt, width=width, height=height, steps=steps, guidance=guidance)
        if out["type"] == "image":
            st.image(out["data"], use_column_width=True)
            buf = io.BytesIO(); out["data"].save(buf, format="PNG"); buf.seek(0)
            st.download_button("Download PNG", data=buf.getvalue(), file_name="pictator_output.png", mime="image/png")
        else:
            st.error(out.get("data", "Unknown HF Router error"))

st.markdown("---")
st.caption("© 2025 Harmony Strategy Partner — Costulator + Pictator Suite (OpenRouter + HF)")
