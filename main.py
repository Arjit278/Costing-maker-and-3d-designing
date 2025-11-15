# Harmony Costulator + Pictator Analyzer (v1.5.4) — AI OCR mode integrated
# Adds: google/gemini-2.5-flash-lite AI OCR primary, OpenRouter fallback.
# Keep other logic unchanged; only AI OCR and robust CSV handling added.

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

# ==== Secrets & Keys ====
# Prefer st.secrets (Streamlit Cloud). Provide sidebar text input fallback for local dev.
def get_secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # no secrets file
        pass
    return default

st.sidebar.header("🔐 API Keys / Overrides (optional for local use)")
openrouter_key = get_secret("OPENROUTER_KEY", None)
hf_token = get_secret("HF_TOKEN", None)
gemini_key = get_secret("GEMINI_KEY", None)
anthropic_key = get_secret("ANTHROPIC_API_KEY", None)

# Allow manual overrides in sidebar for local dev
openrouter_key = st.sidebar.text_input("OpenRouter Key (OpenRouter/OpenAI style)", openrouter_key, type="password")
hf_token = st.sidebar.text_input("HuggingFace Token (HF_TOKEN)", hf_token, type="password")
gemini_key = st.sidebar.text_input("Gemini Key (GEMINI_KEY)", gemini_key, type="password")
anthropic_key = st.sidebar.text_input("Anthropic Key (CLAUDE / ANTHROPIC)", anthropic_key, type="password")

# ==== OpenRouter wrapper (OpenAI-compatible) ====
HAS_OPENAI_SDK = False
try:
    import openai
    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False

def call_openrouter_model(prompt: str, model: str, api_key: str, max_tokens: int = 1024):
    """
    Uses OpenAI-compatible client pointed at OpenRouter as in earlier code.
    Expects api_key and model string e.g. 'meta-llama/llama-4-scout:free' or openrouter-compatible model name.
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
            messages=[
                {"role": "system", "content": "You are a helpful assistant for business and engineering analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[❌ Model call failed: {str(e)}]"

# ==== AI OCR: Gemini (primary) → OpenRouter (fallback) → Tesseract (local fallback) ====
# Gemini via google.generativeai (google/gemini-2.5-flash-lite)
HAS_GOOGLE_GENAI = False
try:
    import google.generativeai as genai
    HAS_GOOGLE_GENAI = True
except Exception:
    HAS_GOOGLE_GENAI = False

def ai_ocr_image_to_text(pil_image: Image.Image, use_gemini_model: str = "gemini-2.5-flash-lite"):
    """
    Try AI OCR:
     1) Gemini (google.generativeai) if gemini_key provided
     2) OpenRouter (OpenAI-compatible) OCR style request if openrouter_key provided
     3) Local tesseract if installed (only for local runs)
    Returns plain text or error string.
    """
    # Prepare image bytes
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # 1) Gemini (google.generativeai)
    if gemini_key and HAS_GOOGLE_GENAI:
        try:
            genai.configure(api_key=gemini_key)
            model = use_gemini_model
            # Use generate_text with image input; Gemini image understanding is provided via "image" in examples.
            # We'll use the "generate" API with multimodal inputs. Exact API may change; attempt robust call.
            response = genai.generate(
                model=model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Extract text and layout details from the image. Return plain text only."},
                        {"type": "input_image", "image_bytes": img_bytes},
                    ]
                }],
                temperature=0.0,
                max_output_tokens=1024
            )
            # Response handling -- genai returns a structure; extract text
            if hasattr(response, "candidates") and response.candidates:
                text = ""
                for c in response.candidates:
                    if getattr(c, "content", None):
                        # content might be a string or list
                        text += str(c.content)
                if text:
                    return text
            # fallback: try attributes
            if getattr(response, "output", None):
                return str(response.output)
        except Exception as e:
            # continue to fallback
            pass

    # 2) OpenRouter via ChatCompletion (ask it to OCR by receiving image base64 as input)
    if openrouter_key and HAS_OPENAI_SDK:
        try:
            openai.api_base = "https://openrouter.ai/api/v1"
            openai.api_key = openrouter_key

            # We encode image in base64 and ask the model to extract text from it.
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            user_msg = (
                "You are an OCR assistant. Extract any readable text from the image "
                "provided below. Only return the text. The image is provided as a base64 string.\n\n"
                f"BASE64_IMAGE:{b64}"
            )
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # try a vision-capable model if available on OpenRouter
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=2048,
            )
            return completion["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 3) Local Tesseract fallback (only when installed)
    if HAS_TESSERACT:
        try:
            text = pytesseract.image_to_string(pil_image)
            return text.strip() or "[No text detected by Tesseract]"
        except Exception:
            pass

    return "[OCR unavailable — no AI key found (Gemini/OpenRouter) and Tesseract not installed]"

# ==== Utilities used previously ====
def enhance_and_ocr(img: Image.Image) -> str:
    """
    If AI OCR mode is enabled we call ai_ocr_image_to_text, else fallback
    to previous enhance pipeline and (preferably) tesseract.
    """
    # Basic image enhancement (sharpen/contrast) to help OCR
    try:
        img_t = img.convert("L")
        img_t = ImageEnhance.Contrast(img_t).enhance(1.6)
        img_t = ImageEnhance.Brightness(img_t).enhance(1.1)
        img_t = img_t.filter(ImageFilter.SHARPEN)
    except Exception:
        img_t = img

    # Use AI OCR
    ai_text = ai_ocr_image_to_text(img_t)
    return ai_text

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
            return "\n".join(enhance_and_ocr(img) for img in images)
        return "[PDF OCR unavailable]"

# ==== Pictator image generation helper (unchanged behavior) ====
def generate_pictator_image(prompt: str, model_choice: str, api_key: str):
    """
    Keep existing logic: tries OpenAI / Claude / Gemini branches
    (we keep this comment for transparency). Uses OpenRouter or native SDKs when available.
    """
    import base64, os
    from io import BytesIO
    from PIL import Image

    try:
        # === OPENAI / GPT-4o-Mini (OpenRouter) branch ===
        if "gpt" in model_choice.lower() or ("openrouter" in model_choice.lower() and openrouter_key):
            if not HAS_OPENAI_SDK:
                return {"type": "error", "data": "[⚠️ OpenAI SDK missing. pip install openai]"}
            try:
                openai.api_base = "https://openrouter.ai/api/v1"
                openai.api_key = api_key
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1024,
                )
                # The OpenRouter ChatCompletion may provide a URL or base64; we return text URL or error
                content = resp["choices"][0]["message"]["content"]
                # If the model returns base64 image content, attempt to decode
                if content.strip().startswith("data:image"):
                    # base64 data URI
                    header, b64 = content.split(",", 1)
                    img_bytes = base64.b64decode(b64)
                    return {"type": "image", "data": Image.open(io.BytesIO(img_bytes))}
                # Else treat as URL or text
                return {"type": "text", "data": content}
            except Exception as e:
                return {"type": "error", "data": f"[❌ OpenRouter generation failed: {e}]"}

        # === CLAUDE Sonnet branch (Anthropic) ===
        elif "claude" in model_choice.lower() and anthropic_key:
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=anthropic_key)
                resp = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4096,
                    temperature=0.3,
                    messages=[{"role": "user",
                               "content": [{"type": "text", "text": prompt}]}],
                )
                for part in resp.content:
                    if getattr(part, "type", None) == "image" and hasattr(part, "image"):
                        return {"type": "url", "data": part.image.url}
                return {"type": "text", "data": resp}
            except Exception as e:
                return {"type": "error", "data": f"[❌ Claude call failed: {e}]"}
        # === GEMINI branch (google.generativeai) ===
        elif "gemini" in model_choice.lower() and gemini_key and HAS_GOOGLE_GENAI:
            try:
                genai.configure(api_key=gemini_key)
                model = model_choice.split()[-1] if " " in model_choice else "gemini-2.5"
                response = genai.generate(
                    model="gemini-2.5-flash-lite",
                    input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                    max_output_tokens=1024,
                    temperature=0.2
                )
                if hasattr(response, "candidates") and response.candidates:
                    return {"type": "text", "data": response.candidates[0].content}
                return {"type": "text", "data": str(response)}
            except Exception as e:
                return {"type": "error", "data": f"[❌ Gemini call failed: {e}]"}
        else:
            return {"type": "error", "data": "[⚠️ Unsupported pictator engine or missing key]"}
    except Exception as e:
        return {"type": "error", "data": f"[Image generation error: {str(e)}]"}

# ==== Streamlit UI ====
st.set_page_config(page_title="Harmony Costulator + Pictator Analyzer", layout="wide")
st.title("⚡ Harmony Costulator + Pictator Analyzer (v1.5.4)")

# keys input reminder
st.sidebar.info("Secrets: you can store OPENROUTER_KEY, HF_TOKEN, GEMINI_KEY in Streamlit Secrets or paste here.")
st.sidebar.write("OpenRouter key present:" , bool(openrouter_key))
st.sidebar.write("Gemini key present:", bool(gemini_key))

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
    enable_ai_ocr = st.checkbox("Enable AI OCR mode (no Tesseract)", True)

    if file:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            tmp = f"temp_{file.name}"
            with open(tmp, "wb") as f:
                f.write(file.read())
            with st.spinner("Extracting text from PDF..."):
                extracted_text = ocr_pdf(tmp)
        else:
            img = Image.open(file)
            st.image(img, caption="Uploaded Drawing", use_container_width=True)
            with st.spinner("Performing OCR on image..."):
                if enable_ai_ocr:
                    extracted_text = ai_ocr_image_to_text(img)
                else:
                    extracted_text = enhance_and_ocr(img)

        st.text_area("📜 Extracted Text", extracted_text, height=180)

        if st.button("🔍 Run Pictator Analysis"):
            st.info("Running Pictator Analyzer...")
            # Use OpenRouter LLM for summarization
            summary = call_openrouter_model(f"Summarize this drawing:\n\n{extracted_text}", "meta-llama/llama-4-scout:free", api_key)
            st.markdown("### 📘 Drawing Summary")
            st.write(summary)

            analysis = call_openrouter_model(f"{custom_prompt}\n\nDrawing text:\n{extracted_text}", "meta-llama/llama-3.3-70b-instruct:free", api_key)
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
    enable_ai_ocr2 = st.checkbox("Enable AI OCR mode for cost images (no Tesseract)", True)

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
            img = Image.open(cost_file)
            st.image(img, caption="Uploaded Costing Image", use_column_width=True)
            if enable_ai_ocr2:
                text_data = ai_ocr_image_to_text(img)
            else:
                text_data = enhance_and_ocr(img)
            df = None

        st.text_area("🧾 Extracted Cost Data / Table", text_data, height=200)

        if st.button("💰 Run Costulator Analysis"):
            st.info("Running Costulator Analysis...")
            summary = call_openrouter_model(f"Summarize costing:\n{text_data}", "deepseek/deepseek-r1-distill-llama-70b:free", api_key)
            st.markdown("### 📊 Cost Summary")
            st.write(summary)

            # Ask LLM to return a revised CSV with projections already included (cost_0_3m, cost_3_6m, cost_6_9m, cost_9_12m)
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
            revised_csv = call_openrouter_model(analysis_prompt, "mistralai/mistral-7b-instruct:free", api_key)

            # If the returned result is an error message, display it
            if revised_csv and revised_csv.startswith("[❌"):
                st.error(revised_csv)
            else:
                # try to parse CSV to dataframe
                try:
                    new_df = pd.read_csv(io.StringIO(revised_csv))
                    st.subheader("✅ Revised Cost Table (AI-generated projections)")
                    st.dataframe(new_df)

                    # Save full revised CSV for downstream tabs
                    st.session_state["revised_csv_text"] = revised_csv
                    st.session_state["revised_df"] = new_df

                    # Also provide download
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

        # create 6-month snapshot (combine 0-3m and 3-6m)
        try:
            # Ensure numeric columns exist
            for c in ["cost_0_3m", "cost_3_6m", "cost_6_9m", "cost_9_12m"]:
                if c not in base_df.columns:
                    st.error(f"Required column {c} not found in revised table.")
                    raise KeyError

            snapshot_6m = base_df.copy()
            snapshot_6m["cost_6m_total"] = snapshot_6m["cost_0_3m"].astype(float) + snapshot_6m["cost_3_6m"].astype(float)

            snapshot_9m = base_df.copy()
            snapshot_9m["cost_9m_total"] = snapshot_9m["cost_0_3m"].astype(float) + snapshot_9m["cost_3_6m"].astype(float) + snapshot_9m["cost_6_9m"].astype(float)

            # Comparison percentage savings vs original baseline (assume original base col is 'cost' or the first numeric col)
            # Find a reasonable original column
            numeric_cols = [c for c in base_df.columns if base_df[c].dtype in [np.float64, np.int64] or c.lower().startswith("cost")]
            baseline_col = None
            if "cost" in base_df.columns:
                baseline_col = "cost"
            elif numeric_cols:
                baseline_col = numeric_cols[0]
            else:
                baseline_col = None

            comp_table = pd.DataFrame()
            comp_table["item"] = base_df.index.astype(str)
            if baseline_col:
                # Build percent savings
                comp_table["baseline_total"] = base_df[baseline_col].astype(float)
                comp_table["6m_total"] = snapshot_6m["cost_6m_total"]
                comp_table["9m_total"] = snapshot_9m["cost_9m_total"]
                comp_table["pct_savings_6m"] = 100.0 * (comp_table["baseline_total"] - comp_table["6m_total"]) / (comp_table["baseline_total"].replace(0, np.nan))
                comp_table["pct_savings_9m"] = 100.0 * (comp_table["baseline_total"] - comp_table["9m_total"]) / (comp_table["baseline_total"].replace(0, np.nan))
            else:
                # Fallback: just show totals
                comp_table["6m_total"] = snapshot_6m["cost_6m_total"]
                comp_table["9m_total"] = snapshot_9m["cost_9m_total"]

            st.subheader("📊 Comparison Table (percentage savings)")
            st.dataframe(comp_table.fillna("N/A"))

            # Export Excel files
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

# === TAB 4: Pictator Creator (unchanged) ===
with tabs[3]:
    st.subheader("Create Engineering Drawing from Text")
    model_map = {
        "Pictator-1 (Gemini 2.5 Pro)": "gemini-2.5-pro",
        "Pictator-2 (Claude Sonnet Vision)": "claude-3-sonnet-20240229",
        "Pictator-3 (GPT-4o-Mini Vision via OpenRouter)": "gpt-4o-mini"
    }
    model_name = st.selectbox("Select Pictator Engine", list(model_map.keys()))
    model_choice = model_map[model_name]

    drawing_prompt = st.text_area(
        "Drawing Prompt (editable)",
        "Create a detailed mechanical engineering drawing of a CNC component showing diameter, hole distances, tolerances, finishing, labels, and scaling specifications."
    )
    if st.button("🎨 Generate Pictator Drawing"):
        with st.spinner(f"Generating drawing using {model_name}..."):
            result = generate_pictator_image(drawing_prompt, model_choice, openrouter_key or anthropic_key or gemini_key)

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
