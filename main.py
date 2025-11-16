# streamlit_app_updated.py
# Harmony Costulator + Pictator Analyzer — Updated Tab1/Tab2 with OpenRouter OCR + analyzer fallbacks
import os
import io
import re
import time
import base64
import json
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st

# === Optional Tesseract ===
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

# === Secrets ===
try:
    OPENROUTER_KEY = st.secrets["OPENROUTER_KEY"]
except Exception:
    OPENROUTER_KEY = None

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = None

st.sidebar.header("Keys")
st.sidebar.write("OpenRouter:", bool(OPENROUTER_KEY))
st.sidebar.write("HF token:", bool(HF_TOKEN))

# === OpenAI SDK (used to call OpenRouter) ===
HAS_OPENAI = False
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# === Provided call_openrouter_model (exact as you gave) ===
def call_openrouter_model(prompt: str, model: str, api_key: str) -> str:
    if not HAS_OPENAI:
        return "[⚠️ OpenAI SDK missing. Run: pip install openai==0.28.0]"
    try:
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a structured business & cost analysis assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[❌ Model call failed: {str(e)}]"

# === Analyzer fallback models (try in order) ===
ANALYZER_MODELS = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-7b-instruct:free",
]

def try_multiple_analyzers(prompt: str, api_key: str, models=ANALYZER_MODELS):
    last_err = None
    for model in models:
        out = call_openrouter_model(prompt, model, api_key)
        if isinstance(out, str) and not out.strip().startswith("[❌"):
            return out, model
        last_err = (model, out)
    return f"[❌ All analyzers failed. Last: {last_err}]", None

# === OCR models via OpenRouter (ordered) ===
OCR_MODELS = [
    "google/gemini-2.5-flash-lite",
    "01-ai/yi-vision"
]

def ai_ocr_image(img: Image.Image, api_key: str):
    """
    Try OCR via OpenRouter models in order. Send base64 PNG in prompt and instruct to return only text.
    Falls back to local Tesseract if available.
    Returns: (text, model_used_or_none)
    """
    # prepare base64 PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    ocr_prompt_template = (
        "You are an OCR assistant. Extract all readable text from the image provided below and return the text only, "
        "no commentary, no labels. The image is provided as a data URI.\n\n"
        "IMAGE_DATA_URI:data:image/png;base64,{b64}"
    )

    if not api_key:
        return "[OCR skipped: OpenRouter key missing]", None

    for model in OCR_MODELS:
        try:
            prompt = ocr_prompt_template.format(b64=b64)
            resp = call_openrouter_model(prompt, model, api_key)
            # if resp isn't an error return it
            if isinstance(resp, str) and not resp.strip().startswith("[❌"):
                return resp.strip(), model
        except Exception:
            continue

    # tesseract fallback
    if HAS_TESSERACT:
        try:
            txt = pytesseract.image_to_string(img)
            return txt.strip() or "[No text found by Tesseract]", "tesseract"
        except Exception:
            pass

    return "[OCR failed: no model returned usable text]", None

# === small helper to extract CSV block from model text ===
def extract_csv_from_text(text: str):
    if not isinstance(text, str):
        return None
    code = re.search(r"```(?:csv|text)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code:
        return code.group(1).strip()
    # otherwise try to find contiguous lines with commas
    lines = text.splitlines()
    for i, L in enumerate(lines):
        if "," in L and len(L.split(",")) > 1:
            j = i
            out = []
            while j < len(lines) and lines[j].strip() != "":
                out.append(lines[j])
                j += 1
            if len(out) >= 2:
                return "\n".join(out).strip()
    if "," in text and "\n" in text:
        return text.strip()
    return None

# === Utility OCR/Enhance (if needed) ===
def enhance_image_for_ocr(img: Image.Image):
    try:
        img2 = img.convert("L")
        img2 = ImageEnhance.Contrast(img2).enhance(1.6)
        img2 = ImageEnhance.Brightness(img2).enhance(1.1)
        img2 = img2.filter(ImageFilter.SHARPEN)
        return img2
    except Exception:
        return img

# === Streamlit UI ===
st.set_page_config(page_title="Harmony Costulator + Pictator Analyzer", layout="wide")
st.title("⚡ Harmony Costulator + Pictator Analyzer (Updated OCR + Analyzer Fallbacks)")

# decide which key to use
api_key = OPENROUTER_KEY

tabs = st.tabs(["🧠 Pictator Analyzer", "📊 Costulator (Profitability)", "📈 Costulator Generator", "🎨 Pictator Creator"])

# -----------------------
# TAB 1: Pictator Analyzer (updated)
# -----------------------
with tabs[0]:
    st.subheader("Upload Drawing / Design Image or PDF")
    upload = st.file_uploader("Upload image or PDF", type=["jpg", "jpeg", "png", "pdf"])
    custom_prompt = st.text_area("Custom Prompt (editable)", "Analyze this engineering drawing for materials, machining process, tooling setup, optimization, and improvements.")
    enable_ai_ocr = st.checkbox("Enable AI OCR mode (no Tesseract)", value=True)

    if upload:
        ext = upload.name.split(".")[-1].lower()
        extracted_text = ""
        ocr_model_used = None

        if ext == "pdf":
            # try convert first page to image if pdf2image available
            try:
                from pdf2image import convert_from_bytes
                pages = convert_from_bytes(upload.read())
                img = pages[0]
                st.image(img, caption="First page (PDF)")
                if enable_ai_ocr and api_key:
                    img_proc = enhance_image_for_ocr(img)
                    extracted_text, ocr_model_used = ai_ocr_image(img_proc, api_key)
                elif HAS_TESSERACT:
                    extracted_text = pytesseract.image_to_string(img)
                else:
                    extracted_text = "[PDF uploaded — OCR not available]"
            except Exception as e:
                extracted_text = f"[PDF processing error: {e}]"
        else:
            img = Image.open(upload).convert("RGB")
            st.image(img, caption="Uploaded drawing", use_column_width=True)
            if enable_ai_ocr and api_key:
                img_proc = enhance_image_for_ocr(img)
                extracted_text, ocr_model_used = ai_ocr_image(img_proc, api_key)
            else:
                if HAS_TESSERACT:
                    extracted_text = pytesseract.image_to_string(img)
                else:
                    extracted_text = "[Image uploaded — OCR not available]"

        st.subheader("📜 Extracted Text (OCR)")
        st.text_area("OCR output", extracted_text, height=200)
        if ocr_model_used:
            st.caption(f"OCR model: {ocr_model_used}")

        if st.button("🔍 Run Pictator Analysis"):
            if not api_key:
                st.error("OpenRouter key missing in st.secrets['OPENROUTER_KEY']")
            else:
                # build short prompt and call analyzers with fallback
                prompt = f"Summarize this drawing and list actionable machining/tooling improvements:\n\n{extracted_text}\n\n{custom_prompt}"
                st.info("Running analyzer (will try fallback models until success)...")
                analysis_text, model_used = try_multiple_analyzers(prompt, api_key)
                st.subheader("📘 Drawing Summary and Recommendations")
                st.write(analysis_text)
                st.caption(f"Analyzer model used: {model_used}")
                st.download_button("Download Analysis (TXT)", data=analysis_text, file_name="pictator_analysis.txt", mime="text/plain")

# -----------------------
# TAB 2: Costulator (updated)
# -----------------------
with tabs[1]:
    st.subheader("Upload Costing Sheet or Report")
    cost_file = st.file_uploader("Upload costing (CSV / image / PDF)", type=["csv", "xls", "xlsx", "jpg", "jpeg", "png", "pdf"])
    cost_prompt = st.text_area("Custom Prompt", "Analyze this costing sheet for profitability and generate a 3–9 month cost optimization plan.")
    enable_ai_ocr_cost = st.checkbox("Enable AI OCR for cost images (no Tesseract)", value=True)

    if cost_file:
        ext = cost_file.name.split(".")[-1].lower()
        df = None
        text_data = ""

        if ext in ("csv",):
            try:
                df = pd.read_csv(cost_file)
                st.subheader("Uploaded table (rendered exactly)")
                st.dataframe(df)
                text_data = df.to_csv(index=False)
            except Exception as e:
                st.error(f"CSV parse error: {e}")
        elif ext in ("xls", "xlsx"):
            try:
                df = pd.read_excel(cost_file)
                st.subheader("Uploaded spreadsheet")
                st.dataframe(df)
                text_data = df.to_csv(index=False)
            except Exception as e:
                st.error(f"Excel parse error: {e}")
        elif ext == "pdf":
            try:
                from pdf2image import convert_from_bytes
                pages = convert_from_bytes(cost_file.read())
                st.image(pages[0], caption="First page (PDF)")
                if enable_ai_ocr_cost and api_key:
                    img_proc = enhance_image_for_ocr(pages[0])
                    text_data, ocr_used = ai_ocr_image(img_proc, api_key)
                    st.text_area("Extracted text (preview)", text_data[:3000], height=200)
                else:
                    text_data = "[PDF uploaded — OCR disabled]"
            except Exception as e:
                text_data = f"[PDF parsing error: {e}]"
                st.error(text_data)
        else:
            # image
            img = Image.open(cost_file).convert("RGB")
            st.image(img, caption="Uploaded cost image", use_column_width=True)
            if enable_ai_ocr_cost and api_key:
                img_proc = enhance_image_for_ocr(img)
                text_data, ocr_used = ai_ocr_image(img_proc, api_key)
                st.text_area("Extracted text (preview)", text_data[:3000], height=200)
            else:
                if HAS_TESSERACT:
                    text_data = pytesseract.image_to_string(img)
                    st.text_area("Tesseract OCR output", text_data[:3000], height=200)
                else:
                    st.error("OCR disabled and Tesseract not available — upload CSV for best results.")

        if st.button("💰 Run Costulator Analysis"):
            if not api_key:
                st.error("OpenRouter key missing in st.secrets['OPENROUTER_KEY']")
            else:
                # 1) human-readable summary using analyzer fallback
                summary_prompt = f"Summarize the costing data and identify cost components, likely savings, and suggested optimizations:\n\n{text_data}\n\n{cost_prompt}"
                st.info("Running human-readable analysis (fallback models)...")
                human_summary, model_used = try_multiple_analyzers(summary_prompt, api_key)
                st.subheader("🔎 Cost Summary")
                st.write(human_summary)
                st.caption(f"Analyzer model used: {model_used}")

                # 2) produce revised CSV with projection columns (CSV-only)
                revised_prompt = (
                    "You are given a cost CSV or text. Produce a revised CSV that preserves all original columns and rows "
                    "and add numeric columns: cost_0_3m, cost_3_6m, cost_6_9m, cost_9_12m. "
                    "Estimate optimized numeric values for each row. Return ONLY the CSV text (no commentary).\n\n"
                    f"{text_data}"
                )
                st.info("Requesting revised CSV (model will be tried in fallback order)...")
                revised_text, model_csv = try_multiple_analyzers(revised_prompt, api_key)
                st.caption(f"CSV-generator model used: {model_csv}")

                if revised_text.startswith("[❌"):
                    st.error("Revised CSV generation failed: " + revised_text)
                else:
                    csv_candidate = extract_csv_from_text(revised_text) or revised_text
                    try:
                        new_df = pd.read_csv(io.StringIO(csv_candidate))
                        st.success("Revised CSV parsed successfully.")
                        st.subheader("✅ Revised Cost Table (AI projections)")
                        st.dataframe(new_df)
                        # persist for Tab 3
                        st.session_state["revised_df"] = new_df
                        st.session_state["original_df"] = df
                        st.session_state["revised_csv_text"] = csv_candidate
                        st.download_button("Download Revised CSV", data=csv_candidate, file_name="revised_costs_full.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Could not parse revised CSV: {e}")
                        st.code(revised_text[:4000])

# -----------------------
# Tabs 3 & 4 - keep as previously (no changes in logic)
# -----------------------
# Tab 3: Costulator Generator — use revised_df stored in session to build 6/9 month snapshots and comparison (same behavior as prior)
with tabs[2]:
    st.subheader("📈 Costulator Generator")
    if "revised_df" not in st.session_state:
        st.info("Run Tab 2 analysis first to generate revised CSV.")
    else:
        base_df = st.session_state["revised_df"].copy()
        original_df = st.session_state.get("original_df", None)
        st.write("Original uploaded table (if available):")
        if original_df is not None:
            st.dataframe(original_df)
        st.write("AI Revised table (with projection columns):")
        st.dataframe(base_df)

        required_cols = ["cost_0_3m", "cost_3_6m", "cost_6_9m", "cost_9_12m"]
        missing = [c for c in required_cols if c not in base_df.columns]
        if missing:
            st.error(f"Revised CSV missing projection columns: {missing}")
        else:
            snap6 = base_df.copy()
            snap6["cost_6m_total"] = snap6["cost_0_3m"].astype(float) + snap6["cost_3_6m"].astype(float)
            snap9 = base_df.copy()
            snap9["cost_9m_total"] = snap9["cost_0_3m"].astype(float) + snap9["cost_3_6m"].astype(float) + snap9["cost_6_9m"].astype(float)

            baseline_col = None
            if "cost" in base_df.columns:
                baseline_col = "cost"
            else:
                num_cols = [c for c in base_df.columns if pd.api.types.is_numeric_dtype(base_df[c])]
                if num_cols:
                    baseline_col = num_cols[0]

            comp = pd.DataFrame()
            comp["row_index"] = base_df.index.astype(str)
            if baseline_col:
                comp["baseline_total"] = base_df[baseline_col].astype(float)
                comp["6m_total"] = snap6["cost_6m_total"]
                comp["9m_total"] = snap9["cost_9m_total"]
                comp["pct_savings_6m"] = 100.0 * (comp["baseline_total"] - comp["6m_total"]) / comp["baseline_total"].replace(0, np.nan)
                comp["pct_savings_9m"] = 100.0 * (comp["baseline_total"] - comp["9m_total"]) / comp["baseline_total"].replace(0, np.nan)
            else:
                comp["6m_total"] = snap6["cost_6m_total"]
                comp["9m_total"] = snap9["cost_9m_total"]

            st.subheader("Comparison & Savings")
            st.dataframe(comp.fillna("N/A"))

            # export Excel w/ sheets
            to6 = io.BytesIO()
            to9 = io.BytesIO()
            with pd.ExcelWriter(to6, engine="xlsxwriter") as writer:
                base_df.to_excel(writer, sheet_name="revised_base", index=False)
                snap6.to_excel(writer, sheet_name="6_month_snapshot", index=False)
                comp.to_excel(writer, sheet_name="comparison", index=False)
            with pd.ExcelWriter(to9, engine="xlsxwriter") as writer:
                base_df.to_excel(writer, sheet_name="revised_base", index=False)
                snap9.to_excel(writer, sheet_name="9_month_snapshot", index=False)
                comp.to_excel(writer, sheet_name="comparison", index=False)
            st.download_button("Download 6-month Excel", data=to6.getvalue(), file_name="costs_6_months.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("Download 9-month Excel", data=to9.getvalue(), file_name="costs_9_months.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Tab 4: Pictator Creator (HF Router image gen) — unchanged behavior; uses HF_TOKEN from secrets
with tabs[3]:
    st.subheader("🎨 Pictator Creator")
    HF_MODELS = {
        "Sketchers (Lineart / Mechanical)": "black-forest-labs/FLUX.1-dev",
        "CAD Drawing XL (2D CNC Blueprints)": "stabilityai/stable-diffusion-xl-base-1.0",
        "RealisticVision (3D Render)": "stabilityai/stable-diffusion-3-medium-diffusers"
    }
    pict_choice = st.selectbox("Choose HF model", list(HF_MODELS.keys()))
    pict_repo = HF_MODELS[pict_choice]
    pict_prompt = st.text_area("Drawing prompt", "technical blueprint lineart of a disc brake, top view, thin lines, engineering drawing")
    c1, c2 = st.columns(2)
    w = c1.number_input("Width", 256, 1536, 768)
    h = c2.number_input("Height", 256, 1536, 768)
    steps = st.slider("Steps", 5, 80, 30)
    guidance = st.slider("Guidance", 1.0, 12.0, 3.5)

    def hf_router_generate_image(hf_token, model_repo, prompt, width=768, height=768, steps=30, guidance=3.5):
        if not hf_token:
            return {"type": "error", "data": "[HF_TOKEN missing in st.secrets]"}
        url = f"https://router.huggingface.co/hf-inference/models/{model_repo}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": prompt, "parameters": {"width": width, "height": height, "num_inference_steps": steps, "guidance_scale": guidance}}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
        except Exception as e:
            return {"type": "error", "data": f"HF Router request failed: {e}"}
        if "image" in r.headers.get("content-type", ""):
            try:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                return {"type": "image", "data": img}
            except Exception as e:
                return {"type": "error", "data": f"Image decode failed: {e}"}
        try:
            data = r.json()
            if isinstance(data, dict) and "generated_image" in data:
                b = base64.b64decode(data["generated_image"])
                return {"type": "image", "data": Image.open(io.BytesIO(b)).convert("RGB")}
            if isinstance(data, dict) and "images" in data and data["images"]:
                b = base64.b64decode(data["images"][0]); return {"type": "image", "data": Image.open(io.BytesIO(b)).convert("RGB")}
            if isinstance(data, list) and data and isinstance(data[0], dict):
                for k in ("blob","generated_image","image"):
                    if k in data[0]:
                        b = base64.b64decode(data[0][k]); return {"type": "image", "data": Image.open(io.BytesIO(b)).convert("RGB")}
            return {"type": "error", "data": f"Unsupported HF response: {str(data)[:300]}"}
        except Exception as e:
            return {"type": "error", "data": f"HF Router non-json response or decode failed: {str(e)}"}

    if st.button("Generate Pictator Drawing"):
        out = hf_router_generate_image(HF_TOKEN, pict_repo, pict_prompt, width=w, height=h, steps=steps, guidance=guidance)
        if out["type"] == "image":
            st.image(out["data"], use_column_width=True)
            b = io.BytesIO(); out["data"].save(b, format="PNG"); b.seek(0)
            st.download_button("Download PNG", data=b.getvalue(), file_name="pictator_output.png", mime="image/png")
        else:
            st.error(out.get("data", "HF Router error"))

st.markdown("---")
st.caption("© 2025 Harmony Strategy Partner — Costulator + Pictator Suite")
