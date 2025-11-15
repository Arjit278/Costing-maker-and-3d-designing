# Harmony Costulator + Pictator Analyzer (v1.5.4) — OpenRouter AI OCR (Gemini primary, Yi-Vision fallback)
# Only OCR pipeline changed — rest logic preserved as requested.

import os
import io
import base64
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st
import requests
import time

# Optional local tesseract support (will not run on Streamlit Cloud).
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

# ==== Secrets & Keys (Streamlit Secrets preferred) ====
def get_secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

st.sidebar.header("🔐 API Keys / Overrides (optional for local use)")
openrouter_key = get_secret("OPENROUTER_KEY", None)
hf_token = get_secret("HF_TOKEN", None)

# Allow manual overrides in sidebar for local dev
openrouter_key = st.sidebar.text_input("OpenRouter Key (OPENROUTER_KEY)", openrouter_key, type="password")
hf_token = st.sidebar.text_input("HuggingFace Token (HF_TOKEN)", hf_token, type="password")

# ==== OpenRouter (OpenAI-compatible) SDK detection ====
HAS_OPENAI_SDK = False
try:
    import openai
    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False

if not HAS_OPENAI_SDK:
    st.sidebar.warning("Install openai: pip install openai — required for OpenRouter calls.")

# ==== OpenRouter helper for text completions and vision OCR ====
def call_openrouter_chat(prompt: str, model: str, api_key: str, max_tokens: int = 1024):
    """
    General helper to call OpenRouter via openai.ChatCompletion (OpenAI-compatible).
    Returns text or an error string starting with [❌ ...]
    """
    if not api_key:
        return "[❌ OpenRouter API key missing]"
    if not HAS_OPENAI_SDK:
        return "[❌ OpenAI SDK not installed: pip install openai]"
    try:
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are an assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[❌ Model call failed: {str(e)}]"

def call_openrouter_vision_ocr(pil_image: Image.Image, api_key: str,
                               primary_model="google/gemini-2.5-flash-lite",
                               fallback_model="01-ai/yi-vision",
                               final_fallback="gpt-4o-mini"):
    """
    Try multi-stage OpenRouter OCR:
      1) Primary model (Gemini 2.5 flash lite) via OpenRouter ChatCompletion with image base64 in prompt
      2) Fallback model Yi-Vision via same approach
      3) final fallback gpt-4o-mini attempt
      4) Local tesseract if available
    Returns extracted text (string) or an error message.
    """
    # prepare base64 image
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_b = buf.getvalue()
    b64 = base64.b64encode(img_b).decode("utf-8")

    if not api_key:
        return "[OCR unavailable — OpenRouter key not provided]"

    if not HAS_OPENAI_SDK:
        return "[OCR unavailable — openai SDK not installed]"

    # prepare message to ask OCR
    # Keep it short and explicit: provide base64, ask for plain text only
    user_msg_template = (
        "You are an OCR assistant. Extract any readable text, table data, and numeric values "
        "from the image provided as a base64 data URI. ONLY return the text/table in plain text or CSV format; "
        "no commentary.\n\nBASE64_IMAGE:data:image/png;base64,{b64}"
    ).format(b64=b64)

    # Try primary
    try:
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=primary_model,
            messages=[{"role":"user","content": user_msg_template}],
            temperature=0.0,
            max_tokens=4096
        )
        out = resp["choices"][0]["message"]["content"]
        # If the model returned an explicit error or empty, try fallback
        if out and not out.strip().startswith("[❌") and len(out.strip())>0:
            return out
    except Exception:
        # fall through to fallback
        pass

    # Try yi-vision (fallback)
    try:
        resp = openai.ChatCompletion.create(
            model=fallback_model,
            messages=[{"role":"user","content": user_msg_template}],
            temperature=0.0,
            max_tokens=4096
        )
        out = resp["choices"][0]["message"]["content"]
        if out and not out.strip().startswith("[❌") and len(out.strip())>0:
            return out
    except Exception:
        pass

    # Final attempt with gpt-4o-mini general vision-capable fallback
    try:
        resp = openai.ChatCompletion.create(
            model=final_fallback,
            messages=[{"role":"user","content": user_msg_template}],
            temperature=0.0,
            max_tokens=4096
        )
        out = resp["choices"][0]["message"]["content"]
        if out and not out.strip().startswith("[❌") and len(out.strip())>0:
            return out
    except Exception:
        pass

    # Local tesseract fallback
    if HAS_TESSERACT:
        try:
            text = pytesseract.image_to_string(pil_image)
            return text.strip() or "[No text detected by Tesseract]"
        except Exception:
            pass

    return "[OCR unavailable — all AI OCR attempts failed and Tesseract not installed]"

# ==== Utilities used previously ====
def enhance_and_ocr(img: Image.Image) -> str:
    """
    Image enhancement + attempt local tesseract (if allowed).
    If you want AI OCR you should call call_openrouter_vision_ocr directly.
    """
    try:
        img_t = img.convert("L")
        img_t = ImageEnhance.Contrast(img_t).enhance(1.6)
        img_t = ImageEnhance.Brightness(img_t).enhance(1.1)
        img_t = img_t.filter(ImageFilter.SHARPEN)
    except Exception:
        img_t = img

    if HAS_TESSERACT:
        try:
            out = pytesseract.image_to_string(img_t)
            return out.strip() or "[No text detected by Tesseract]"
        except Exception:
            return "[Tesseract OCR failed]"
    return "[No OCR available (Tesseract not installed)]"

def ocr_pdf(path: str) -> str:
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip() or "[No text found in PDF]"
    except Exception:
        if convert_from_path:
            images = convert_from_path(path)
            # Use AI OCR on each page if openrouter_key present, else tesseract
            texts = []
            for p in images:
                if openrouter_key:
                    texts.append(call_openrouter_vision_ocr(p, openrouter_key))
                else:
                    texts.append(enhance_and_ocr(p))
            return "\n".join(texts)
        return "[PDF OCR unavailable]"

# ==== Pictator image generation helper (kept compatible) ====
def generate_pictator_image(prompt: str, api_key: str, model_choice: str = "gpt-4o-mini"):
    """
    Use OpenRouter ChatCompletion to request an image or base64 blob from a vision model.
    We attempt to use the OpenRouter chat completion with the provided model_choice (if available).
    Returns dict with type: "image"/"text"/"error"
    """
    if not api_key:
        return {"type":"error", "data":"[OpenRouter key missing]"}
    if not HAS_OPENAI_SDK:
        return {"type":"error", "data":"[openai SDK missing]"}

    try:
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key
        # We ask the model to return base64 PNG image as data URI if capable
        user_prompt = (
            "You are an image generation assistant. Given the prompt below, return only a PNG image "
            "encoded as a data URI (data:image/png;base64,...) with no additional text if you can generate images. "
            "If you cannot return images, return a short text explaining how to produce the image.\n\n"
            f"PROMPT:\n{prompt}"
        )
        resp = openai.ChatCompletion.create(
            model=model_choice,
            messages=[{"role":"user","content":user_prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        content = resp["choices"][0]["message"]["content"]
        if not content:
            return {"type":"error","data":"[No output from model]"}
        content = content.strip()

        if content.startswith("data:image"):
            # decode and return PIL image
            header, b64 = content.split(",", 1)
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes))
            return {"type":"image","data": img}
        else:
            # may be URL, or textual instructions
            return {"type":"text","data":content}
    except Exception as e:
        return {"type":"error","data":f"[OpenRouter generation failed: {e}]"}

# ==== Streamlit UI & Tabs (unchanged logic, only OCR calls replaced) ====
st.set_page_config(page_title="Harmony Costulator + Pictator Analyzer", layout="wide")
st.title("⚡ Harmony Costulator + Pictator Analyzer (v1.5.4)")

st.sidebar.info("Store OPENROUTER_KEY, HF_TOKEN in Streamlit Secrets, or paste them here for local testing.")
st.sidebar.write("OpenRouter key present:", bool(openrouter_key))
st.sidebar.write("HF token present:", bool(hf_token))

api_key = openrouter_key  # used for OpenRouter model calls in this app
if not api_key:
    st.warning("OpenRouter API key not provided — OpenRouter-dependent features will not work.")

tabs = st.tabs(["🧠 Pictator Analyzer", "📊 Costulator (Profitability)", "📈 Costulator Generator", "🎨 Pictator Creator"])

# === TAB 1: Pictator Analyzer ===
with tabs[0]:
    st.subheader("Upload Drawing / Design Image or PDF")
    file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])
    custom_prompt = st.text_area(
        "Custom Prompt (editable)",
        "Analyze this engineering drawing for materials, machining process, tooling setup, optimization, and improvements."
    )
    enable_ai_ocr = st.checkbox("Enable AI OCR mode (OpenRouter - Gemini primary, Yi-Vision fallback)", True)

    if file:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            tmp = f"temp_{file.name}"
            with open(tmp, "wb") as f:
                f.write(file.read())
            with st.spinner("Extracting text from PDF..."):
                extracted_text = ocr_pdf(tmp)
        else:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Uploaded Drawing", use_container_width=True)
            with st.spinner("Performing OCR on image..."):
                if enable_ai_ocr and api_key:
                    extracted_text = call_openrouter_vision_ocr(img, api_key,
                                                               primary_model="google/gemini-2.5-flash-lite",
                                                               fallback_model="01-ai/yi-vision",
                                                               final_fallback="gpt-4o-mini")
                else:
                    extracted_text = enhance_and_ocr(img)

        st.text_area("📜 Extracted Text", extracted_text, height=180)

        if st.button("🔍 Run Pictator Analysis"):
            st.info("Running Pictator Analyzer...")
            summary = call_openrouter_chat(f"Summarize this drawing:\n\n{extracted_text}", "meta-llama/llama-4-scout:free", api_key)
            st.markdown("### 📘 Drawing Summary")
            st.write(summary)

            analysis = call_openrouter_chat(f"{custom_prompt}\n\nDrawing text:\n{extracted_text}", "meta-llama/llama-3.3-70b-instruct:free", api_key)
            st.subheader("🧩 Pictator AI Engineering Insights")
            st.write(analysis)
            st.download_button("⬇️ Download Pictator Analysis", data=analysis, file_name="pictator_analysis.txt")

# === TAB 2: Costulator (Profitability) ===
with tabs[1]:
    st.subheader("Upload Costing Sheet or Report (CSV preferred)")
    cost_file = st.file_uploader("Upload costing image, CSV, or PDF", type=["csv", "jpg", "jpeg", "png", "pdf"])
    cost_prompt = st.text_area(
        "Custom Prompt",
        "Analyze this costing sheet for profitability and generate a 3–9 month cost optimization plan."
    )
    enable_ai_ocr2 = st.checkbox("Enable AI OCR mode for cost images (OpenRouter)", True)

    if cost_file:
        ext = cost_file.name.split(".")[-1].lower()
        if ext == "csv":
            df = pd.read_csv(cost_file)
            st.dataframe(df)
            text_data = df.to_string(index=False)
        elif ext == "pdf":
            tmp = f"temp_{cost_file.name}"
            with open(tmp, "wb") as f:
                f.write(cost_file.read())
            text_data = ocr_pdf(tmp)
            df = None
        else:
            img = Image.open(cost_file).convert("RGB")
            st.image(img, caption="Uploaded Costing Image", use_column_width=True)
            if enable_ai_ocr2 and api_key:
                text_data = call_openrouter_vision_ocr(img, api_key,
                                                      primary_model="google/gemini-2.5-flash-lite",
                                                      fallback_model="01-ai/yi-vision",
                                                      final_fallback="gpt-4o-mini")
            else:
                text_data = enhance_and_ocr(img)
            df = None

        st.text_area("🧾 Extracted Cost Data / Table", text_data, height=200)

        if st.button("💰 Run Costulator Analysis"):
            st.info("Running Costulator Analysis...")
            summary = call_openrouter_chat(f"Summarize costing:\n{text_data}", "deepseek/deepseek-r1-distill-llama-70b:free", api_key)
            st.markdown("### 📊 Cost Summary")
            st.write(summary)

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
            revised_csv = call_openrouter_chat(analysis_prompt, "mistralai/mistral-7b-instruct:free", api_key)

            if revised_csv and revised_csv.startswith("[❌"):
                st.error(revised_csv)
            else:
                try:
                    new_df = pd.read_csv(io.StringIO(revised_csv))
                    st.subheader("✅ Revised Cost Table (AI-generated projections)")
                    st.dataframe(new_df)

                    st.session_state["revised_csv_text"] = revised_csv
                    st.session_state["revised_df"] = new_df

                    st.download_button("Download Revised CSV", data=revised_csv, file_name="revised_costs_full.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Could not parse revised CSV from model. Raw output shown below. Error: {e}")
                    st.code(revised_csv or "No output from model")

# === TAB 3: Costulator Generator (Auto Forecast & Excel Snapshots) ===
with tabs[2]:
    st.subheader("📈 Generate Forecasted Cost Sheets & Comparison")
    st.write("This will produce 6-month and 9-month Excel snapshots and a comparison table of percentage savings.")
    if "revised_df" not in st.session_state:
        st.info("Please run Tab 2 Costulator Analysis first to generate revised cost table.")
    else:
        base_df = st.session_state["revised_df"]

        try:
            for c in ["cost_0_3m", "cost_3_6m", "cost_6_9m", "cost_9_12m"]:
                if c not in base_df.columns:
                    st.error(f"Required column {c} not found in revised table.")
                    raise KeyError

            snapshot_6m = base_df.copy()
            snapshot_6m["cost_6m_total"] = snapshot_6m["cost_0_3m"].astype(float) + snapshot_6m["cost_3_6m"].astype(float)

            snapshot_9m = base_df.copy()
            snapshot_9m["cost_9m_total"] = snapshot_9m["cost_0_3m"].astype(float) + snapshot_9m["cost_3_6m"].astype(float) + snapshot_9m["cost_6_9m"].astype(float)

            numeric_cols = [c for c in base_df.columns if pd.api.types.is_numeric_dtype(base_df[c]) or c.lower().startswith("cost")]
            baseline_col = "cost" if "cost" in base_df.columns else (numeric_cols[0] if numeric_cols else None)

            comp_table = pd.DataFrame()
            comp_table["item"] = base_df.index.astype(str)
            if baseline_col:
                comp_table["baseline_total"] = base_df[baseline_col].astype(float)
                comp_table["6m_total"] = snapshot_6m["cost_6m_total"]
                comp_table["9m_total"] = snapshot_9m["cost_9m_total"]
                comp_table["pct_savings_6m"] = 100.0 * (comp_table["baseline_total"] - comp_table["6m_total"]) / (comp_table["baseline_total"].replace(0, np.nan))
                comp_table["pct_savings_9m"] = 100.0 * (comp_table["baseline_total"] - comp_table["9m_total"]) / (comp_table["baseline_total"].replace(0, np.nan))
            else:
                comp_table["6m_total"] = snapshot_6m["cost_6m_total"]
                comp_table["9m_total"] = snapshot_9m["cost_9m_total"]

            st.subheader("📊 Comparison Table (percentage savings)")
            st.dataframe(comp_table.fillna("N/A"))

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

            st.download_button("Download 6-month Excel", data=towrite_6.getvalue(), file_name="costs_6_months.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("Download 9-month Excel", data=towrite_9.getvalue(), file_name="costs_9_months.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Could not build snapshots: {e}")

# === TAB 4: Pictator Creator (keeps functionality but uses OpenRouter for generation) ===
with tabs[3]:
    st.subheader("Create Engineering Drawing from Text")
    # If you want to use specific HF models for pictator-generator, you can call HF Router separately.
    # Here we'll offer OpenRouter options (GPT vision) and a note to use HF models via the HF token elsewhere.
    model_map = {
        "Pictator-OpenRouter (gpt-4o-mini vision)": "gpt-4o-mini"
    }
    model_name = st.selectbox("Select Pictator Engine", list(model_map.keys()))
    model_choice = model_map[model_name]

    drawing_prompt = st.text_area(
        "Drawing Prompt (editable)",
        "Create a detailed mechanical engineering drawing of a CNC component showing diameter, hole distances, tolerances, finishing, labels, and scaling specifications."
    )
    if st.button("🎨 Generate Pictator Drawing"):
        with st.spinner(f"Generating drawing using {model_name}..."):
            result = generate_pictator_image(drawing_prompt, api_key, model_choice)

        if result["type"] == "image":
            st.image(result["data"], caption=f"Generated by {model_name}", use_container_width=True)
            buf = io.BytesIO()
            result["data"].save(buf, format="PNG")
            st.session_state["pictator_image"] = buf.getvalue()
            st.download_button("⬇️ Download Drawing (PNG)", data=st.session_state["pictator_image"], file_name="pictator_generated.png", mime="image/png")
        elif result["type"] == "text":
            st.write(result["data"])
        elif result["type"] == "error":
            st.error(result["data"])
        else:
            st.warning("No usable output returned.")

st.markdown("---")
st.caption("© 2025 Harmony Strategy Partner — Costulator + Pictator Suite (AI OCR enabled)")
