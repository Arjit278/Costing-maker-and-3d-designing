# streamlit_app.py
# Harmony — Costulator + Pictator (Streamlit Cloud ready)
# OCR & analysis: OpenRouter (primary: 01-ai/yi-vision, fallback: google/gemini-2.5-flash-lite)
# Image generation: Hugging Face Router (HF_TOKEN from secrets)

import os
import io
import base64
import time
import json
import pandas as pd
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st

# --- Try to import OpenAI SDK for OpenRouter usage ---
HAS_OPENAI_SDK = False
try:
    import openai
    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False

# --- Config / Secrets ---
st.set_page_config(page_title="Harmony Costulator + Pictator (Streamlit Cloud)", layout="wide")
st.title("⚡ Harmony Costulator + Pictator Analyzer (Streamlit Cloud)")

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None

OPENROUTER_KEY = get_secret("OPENROUTER_KEY")  # required
HF_TOKEN = get_secret("HF_TOKEN")              # required for pictator image generation

# Also allow sidebar overrides for local dev
st.sidebar.header("API keys (optional override)")
openrouter_key_input = st.sidebar.text_input("OpenRouter Key", OPENROUTER_KEY or "", type="password")
hf_token_input = st.sidebar.text_input("HF_TOKEN", HF_TOKEN or "", type="password")

OPENROUTER_KEY = openrouter_key_input.strip() or OPENROUTER_KEY
HF_TOKEN = hf_token_input.strip() or HF_TOKEN

if not OPENROUTER_KEY:
    st.sidebar.error("OPENROUTER_KEY not found in secrets. Add it to App Settings > Secrets.")
if not HF_TOKEN:
    st.sidebar.warning("HF_TOKEN not found in secrets. HF image generation will be disabled until set.")

# --- Utility: image enhancement helper for OCR ---
def enhance_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    try:
        img = pil_img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(1.7)
        img = ImageEnhance.Sharpness(img).enhance(1.2)
        img = ImageEnhance.Brightness(img).enhance(1.05)
        return img
    except Exception:
        return pil_img

# --- OpenRouter helper (ChatCompletion-based) ---
def call_openrouter_chat(model: str, messages, max_tokens=1500, temperature=0.0):
    """
    Uses openai SDK configured to point to openrouter.ai.
    Returns the assistant text or raises an Exception.
    """
    if not HAS_OPENAI_SDK:
        raise RuntimeError("OpenAI SDK not available. Install `openai` in requirements.")
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_KEY not provided in secrets or sidebar.")
    openai.api_key = OPENROUTER_KEY
    openai.api_base = "https://openrouter.ai/api/v1"
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return completion["choices"][0]["message"]["content"]

# --- AI OCR via OpenRouter: primary yi-vision, fallback gemini ---
def ai_ocr_openrouter(pil_img: Image.Image, want_csv: bool = False):
    """
    Returns extracted text or CSV text.
    Primary: model '01-ai/yi-vision'
    Fallback: 'google/gemini-2.5-flash-lite'
    When want_csv=True, instruct the model to output CSV preserving columns when possible.
    """
    # enhance
    img_for_ocr = enhance_image_for_ocr(pil_img)
    buf = io.BytesIO()
    img_for_ocr.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    instruction = (
        "You are an OCR assistant specialized in engineering/cost tables. "
        "Extract all table data and text from the image provided as base64. "
        "If the image contains a table, return CSV text only (no commentary) that preserves the original column names and row order. "
        "If no table, return plain extracted text. "
        "Always return machine-parseable CSV when a table is detected."
    )
    if want_csv:
        instruction = (
            "You are an OCR assistant specialized in engineering/cost tables. "
            "Extract any tabular data and return CSV text only (no commentary). "
            "Preserve original column headers as closely as possible. "
            "If you cannot find a table, return a simple CSV with one column 'extracted_text' and the extracted text as a single row."
        )

    messages = [
        {"role": "system", "content": "You are an accurate OCR assistant for engineering drawings and cost sheets."},
        {"role": "user", "content": instruction},
        {"role": "user", "content": f"IMAGE_BASE64:{b64}"}
    ]

    # Try primary
    for model in ("01-ai/yi-vision", "google/gemini-2.5-flash-lite"):
        try:
            # OpenRouter supports chat completions; some vision models may return a long text with CSV
            res = call_openrouter_chat(model=model, messages=messages, max_tokens=2500, temperature=0.0)
            # Heuristic: if looks like CSV (has commas and newlines and at least one header row), return directly.
            if isinstance(res, str):
                txt = res.strip()
                # If model returned a JSON or quoted block, try to strip markdown fences
                if txt.startswith("```") and txt.endswith("```"):
                    txt = "\n".join(txt.splitlines()[1:-1])
                # If it is CSV-like:
                if ("," in txt and "\n" in txt) or txt.count("\n") >= 1:
                    return txt
                # if it returned plain text, return that in CSV wrapper when want_csv
                if want_csv:
                    # return as single column CSV
                    safe_text = txt.replace('"', '""')
                    return f"extracted_text\n\"{safe_text}\""
                return txt
        except Exception as e:
            # try fallback model
            continue

    # If both fail
    raise RuntimeError("AI OCR failed (OpenRouter models). Check key/permissions or model availability.")

# --- HF Router image generation helper (Pictator) ---
def hf_router_generate_image(model_repo: str, prompt: str, width=768, height=768, steps=30, guidance=3.5):
    """
    Calls HF Router endpoint for the specified model repo.
    Returns PIL.Image on success or raises Exception with message.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set in secrets or sidebar")
    url = f"https://router.huggingface.co/hf-inference/models/{model_repo}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": int(width),
            "height": int(height),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance)
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        # Try to include router message for debugging
        raise RuntimeError(f"HF Router error {resp.status_code}: {resp.text[:1000]}")
    ct = resp.headers.get("content-type", "")
    # If binary image returned:
    if "image" in ct:
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    # Else try JSON candidates
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"HF Router returned non-JSON and not image. Raw: {resp.text[:800]}")
    # Try multiple JSON shapes
    if isinstance(data, dict):
        for key in ("generated_image", "image"):
            if key in data:
                img_b = base64.b64decode(data[key])
                return Image.open(io.BytesIO(img_b)).convert("RGB")
        if "images" in data and isinstance(data["images"], list) and data["images"]:
            img_b = base64.b64decode(data["images"][0])
            return Image.open(io.BytesIO(img_b)).convert("RGB")
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        for key in ("blob", "generated_image", "image"):
            if key in data[0]:
                img_b = base64.b64decode(data[0][key])
                return Image.open(io.BytesIO(img_b)).convert("RGB")
    raise RuntimeError("HF Router returned unknown image format.")

# --- Tab layout ---
tab1, tab2, tab3, tab4 = st.tabs(["🧠 Pictator Analyzer", "📊 Costulator (Profitability)", "📈 Costulator Generator", "🎨 Pictator Creator"])

# -----------------------------
# Tab1: Pictator Analyzer (image upload + OCR + analysis)
# -----------------------------
with tab1:
    st.header("🧠 Pictator Analyzer")
    st.write("Upload a drawing (image/pdf). AI OCR (01-ai/yi-vision primary, gemini fallback) extracts text/tables. OpenRouter LLM analyzes the drawing.")
    file = st.file_uploader("Upload drawing (jpg/png/pdf)", type=["jpg", "jpeg", "png", "pdf"])
    custom_prompt = st.text_area("Analysis prompt (optional)", "Analyze this engineering drawing: materials, machining steps, tooling, tolerances, optimizations.")
    enable_ai_ocr = st.checkbox("Enable AI OCR mode (uses OpenRouter)", True)

    if file:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            # simple PDF -> images extraction fallback using pdf2image if present
            try:
                from pdf2image import convert_from_bytes
                pages = convert_from_bytes(file.read())
                st.image(pages[0], caption="First page preview", use_column_width=True)
                img_for_ocr = pages[0]
            except Exception:
                st.error("PDF processing unavailable in this environment.")
                img_for_ocr = None
        else:
            img_for_ocr = Image.open(file).convert("RGB")
            st.image(img_for_ocr, caption="Uploaded image", use_column_width=True)

        if enable_ai_ocr and img_for_ocr is not None:
            with st.spinner("Running AI OCR (OpenRouter) to extract text/table..."):
                try:
                    ocr_csv_text = ai_ocr_openrouter(img_for_ocr, want_csv=True)
                    st.success("AI OCR completed.")
                    st.download_button("Download OCR CSV", data=ocr_csv_text, file_name="ocr_extracted.csv", mime="text/csv")
                    st.text_area("OCR (CSV or extracted text)", ocr_csv_text, height=200)
                except Exception as e:
                    st.error(f"AI OCR failed: {e}")
                    ocr_csv_text = None

        else:
            ocr_csv_text = None

        if st.button("🔍 Run drawing analysis (OpenRouter)"):
            # Use extracted text (prefer CSV) or image -> ask LLM to analyze
            analysis_input = ""
            if ocr_csv_text:
                analysis_input = ocr_csv_text
            elif img_for_ocr is not None:
                # send short prompt asking to summarize visually
                analysis_input = "Image uploaded. Please analyze visually: " + custom_prompt
            else:
                st.error("No usable input for analysis.")
                analysis_input = None

            if analysis_input:
                with st.spinner("Calling OpenRouter LLM for analysis..."):
                    try:
                        summary = call_openrouter_chat(
                            model="meta-llama/llama-4-scout:free",
                            messages=[
                                {"role": "system", "content": "You are an engineering assistant."},
                                {"role": "user", "content": f"Analyze this drawing or data and summarize: \n\n{analysis_input}\n\nReturn a concise technical summary with suggestions for machining, materials, and optimizations."}
                            ],
                            max_tokens=800,
                            temperature=0.2
                        )
                        st.subheader("📘 Drawing Summary")
                        st.write(summary)
                        # Analysis (detailed)
                        analysis = call_openrouter_chat(
                            model="meta-llama/llama-3.3-70b-instruct:free",
                            messages=[
                                {"role": "system", "content": "You are an engineering & manufacturing analyst."},
                                {"role": "user", "content": f"{custom_prompt}\n\nInput data:\n{analysis_input}"}
                            ],
                            max_tokens=1200,
                            temperature=0.2
                        )
                        st.subheader("🧩 Detailed Analysis")
                        st.write(analysis)
                        st.download_button("Download Pictator Analysis", data=analysis, file_name="pictator_analysis.txt", mime="text/plain")
                    except Exception as e:
                        st.error(f"OpenRouter LLM call failed: {e}")

# -----------------------------
# Tab2: Costulator (Profitability)
# -----------------------------
with tab2:
    st.header("📊 Costulator (Profitability)")
    st.write("Upload CSV / image / PDF costing sheet. If image/PDF, AI OCR will extract table as CSV using OpenRouter vision models (01-ai/yi-vision → google/gemini-2.5-flash-lite fallback).")

    cost_file = st.file_uploader("Upload costing file (CSV / image / PDF)", type=["csv", "jpg", "jpeg", "png", "pdf"])
    cost_custom_prompt = st.text_area("Cost analysis prompt (optional)",
                                     "Analyze the cost sheet and propose optimizations and a 3–9 month cost reduction plan. Return a revised CSV that preserves original columns and adds cost_0_3m, cost_3_6m, cost_6_9m, cost_9_12m.")

    if cost_file:
        ext = cost_file.name.split(".")[-1].lower()
        extracted_text_or_csv = None
        parsed_df = None

        if ext == "csv":
            try:
                parsed_df = pd.read_csv(cost_file)
                extracted_text_or_csv = parsed_df.to_csv(index=False)
                st.success("CSV uploaded and parsed.")
                st.dataframe(parsed_df)
            except Exception as e:
                st.error(f"CSV parse error: {e}")
        else:
            # Image or PDF -> AI OCR to CSV
            with st.spinner("Running AI OCR to extract table as CSV..."):
                try:
                    if ext == "pdf":
                        from pdf2image import convert_from_bytes
                        pages = convert_from_bytes(cost_file.read())
                        img0 = pages[0]
                    else:
                        img0 = Image.open(cost_file).convert("RGB")
                    ocr_csv_text = ai_ocr_openrouter(img0, want_csv=True)
                    extracted_text_or_csv = ocr_csv_text
                    st.success("AI OCR extracted CSV/text.")
                    st.download_button("Download OCR CSV", data=ocr_csv_text, file_name="ocr_costs.csv", mime="text/csv")
                    # Try parse to dataframe
                    try:
                        parsed_df = pd.read_csv(io.StringIO(ocr_csv_text))
                        st.dataframe(parsed_df)
                    except Exception as e:
                        st.warning(f"Could not parse OCR CSV into table: {e}\nShowing raw OCR output below.")
                        st.text_area("OCR raw output", ocr_csv_text, height=200)
                except Exception as e:
                    st.error(f"AI OCR failed: {e}")

        # If parsed_df exists, we can analyze and create revised CSV via LLM
        if parsed_df is not None and st.button("💰 Run Costulator Analysis"):
            with st.spinner("Calling OpenRouter LLM to summarize and produce revised CSV..."):
                # 1) summary
                try:
                    summary = call_openrouter_chat(
                        model="deepseek/deepseek-r1-distill-llama-70b:free",
                        messages=[
                            {"role": "system", "content": "You are a cost analyst."},
                            {"role": "user", "content": f"Summarize this cost table and propose optimizations. Table CSV:\n\n{parsed_df.to_csv(index=False)}"}
                        ],
                        max_tokens=800
                    )
                    st.subheader("📊 Cost Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Summary model call failed: {e}")
                    summary = ""

                # 2) ask LLM to return revised CSV with 4 new columns
                analysis_prompt = f"""
You are given the following CSV (preserve headers and columns exactly). Produce a CSV only (no commentary) that:

- Preserves the original columns and order.
- Adds these new numeric columns: cost_0_3m, cost_3_6m, cost_6_9m, cost_9_12m.
- Each new column should be numeric estimates of the costs after optimization/savings.
- Provide reasonable, explained numbers implied by the analysis. Keep row count identical.

Original CSV:
{parsed_df.to_csv(index=False)}

Return ONLY the CSV text (no explanation).
"""
                try:
                    revised_csv_text = call_openrouter_chat(
                        model="mistralai/mistral-7b-instruct:free",
                        messages=[
                            {"role": "system", "content": "You are an accurate CSV transformer and financial forecaster."},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        max_tokens=1800,
                        temperature=0.2
                    )
                except Exception as e:
                    st.error(f"Revised CSV generation failed: {e}")
                    revised_csv_text = None

                if revised_csv_text:
                    # Try parse revised CSV
                    try:
                        revised_df = pd.read_csv(io.StringIO(revised_csv_text))
                        st.subheader("✅ Revised Cost Table (AI-generated)")
                        st.dataframe(revised_df)
                        # store in session_state for Tab3
                        st.session_state["revised_csv_text"] = revised_csv_text
                        st.session_state["revised_df"] = revised_df
                        st.download_button("Download Revised CSV", data=revised_csv_text, file_name="revised_costs_full.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Could not parse revised CSV returned by model: {e}")
                        st.code(revised_csv_text[:2000])

# -----------------------------
# Tab3: Costulator Generator (6m/9m snapshots & Excel)
# -----------------------------
with tab3:
    st.header("📈 Costulator Generator — 6 & 9 month snapshots")
    st.write("This uses the revised CSV created in Tab 2. It will create two Excel files: 6-month and 9-month snapshots plus a comparison table (percentage savings).")

    if "revised_df" not in st.session_state:
        st.info("Please run Tab 2 and generate the revised CSV first.")
    else:
        base_df = st.session_state["revised_df"].copy()
        st.subheader("Source Revised Table")
        st.dataframe(base_df)

        # Validate required new columns
        required_cols = ["cost_0_3m", "cost_3_6m", "cost_6_9m", "cost_9_12m"]
        missing = [c for c in required_cols if c not in base_df.columns]
        if missing:
            st.error(f"Revised table missing required columns: {missing}")
        else:
            try:
                # create snapshot totals
                base_df["cost_6m_total"] = base_df["cost_0_3m"].astype(float) + base_df["cost_3_6m"].astype(float)
                base_df["cost_9m_total"] = base_df["cost_0_3m"].astype(float) + base_df["cost_3_6m"].astype(float) + base_df["cost_6_9m"].astype(float)

                # Determine baseline column to compare — prefer 'cost' or first numeric
                baseline_col = None
                if "cost" in base_df.columns:
                    baseline_col = "cost"
                else:
                    numeric_cols = [c for c in base_df.columns if np.issubdtype(base_df[c].dtype, np.number)]
                    baseline_col = numeric_cols[0] if numeric_cols else None

                comp_table = pd.DataFrame()
                comp_table["row_id"] = base_df.index.astype(str)
                if baseline_col:
                    comp_table["baseline_total"] = base_df[baseline_col].astype(float)
                    comp_table["6m_total"] = base_df["cost_6m_total"]
                    comp_table["9m_total"] = base_df["cost_9m_total"]
                    comp_table["pct_savings_6m"] = 100.0 * (comp_table["baseline_total"] - comp_table["6m_total"]) / comp_table["baseline_total"].replace(0, np.nan)
                    comp_table["pct_savings_9m"] = 100.0 * (comp_table["baseline_total"] - comp_table["9m_total"]) / comp_table["baseline_total"].replace(0, np.nan)
                else:
                    comp_table["6m_total"] = base_df["cost_6m_total"]
                    comp_table["9m_total"] = base_df["cost_9m_total"]

                st.subheader("Comparison Table (percentage savings)")
                st.dataframe(comp_table.fillna("N/A"))

                # Build Excel files
                towrite_6 = io.BytesIO()
                towrite_9 = io.BytesIO()
                with pd.ExcelWriter(towrite_6, engine="xlsxwriter") as writer:
                    # Save 6-month snapshot sheet plus comparison
                    base_df.to_excel(writer, sheet_name="revised_table", index=False)
                    comp_table.to_excel(writer, sheet_name="comparison", index=False)
                with pd.ExcelWriter(towrite_9, engine="xlsxwriter") as writer:
                    base_df.to_excel(writer, sheet_name="revised_table", index=False)
                    comp_table.to_excel(writer, sheet_name="comparison", index=False)
                towrite_6.seek(0); towrite_9.seek(0)

                st.download_button("Download 6-month Excel", data=towrite_6.getvalue(), file_name="costs_6_months.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.download_button("Download 9-month Excel", data=towrite_9.getvalue(), file_name="costs_9_months.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Could not produce snapshots: {e}")

# -----------------------------
# Tab4: Pictator Creator (HF image generation)
# -----------------------------
with tab4:
    st.header("🎨 Pictator Creator")
    st.write("Use HF Router models to generate engineering drawings (lineart / 3D renders). HF token is read from secrets.")

    MODELS = {
        "Sketchers (Lineart / Mechanical) - FLUX": "black-forest-labs/FLUX.1-dev",
        "CAD Drawing XL (2D CNC Blueprints) - SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
        "RealisticVision (3D Render) - SD3": "stabilityai/stable-diffusion-3-medium-diffusers"
    }

    model_choice = st.selectbox("Choose HF model", list(MODELS.keys()))
    prompt = st.text_area("Drawing prompt:", "technical CNC blueprint lineart of disc brake, top view, thin black lines, engineering drawing")
    col1, col2 = st.columns(2)
    with col1:
        w = st.number_input("Width", 256, 1536, 768)
    with col2:
        h = st.number_input("Height", 256, 1536, 768)
    steps = st.slider("Inference steps", 10, 50, 30)
    guidance = st.slider("Guidance scale", 1.0, 10.0, 3.5)

    if st.button("Generate Pictator Image"):
        model_repo = MODELS[model_choice]
        with st.spinner("Calling HF Router..."):
            try:
                img = hf_router_generate_image(model_repo, prompt, width=w, height=h, steps=steps, guidance=guidance)
                st.image(img, caption="Generated image", use_column_width=True)
                bio = io.BytesIO()
                img.save(bio, format="PNG")
                bio.seek(0)
                st.download_button("Download PNG", data=bio.getvalue(), file_name="pictator_output.png", mime="image/png")
            except Exception as e:
                st.error(f"HF image generation failed: {e}")

st.markdown("---")
st.caption("© 2025 Harmony Strategy Partner — Costulator + Pictator Suite (OpenRouter + HF Router)")
