# streamlit_app.py  (Updated: Integrated, robust, HF Router fallback + OpenRouter UI)
import os
import io
import base64
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st
import requests
import json
import time

# === OCR Support ===
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

def enhance_and_ocr(img: Image.Image) -> str:
    if not HAS_TESSERACT:
        return "[OCR unavailable — simulated text]"
    try:
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Brightness(img).enhance(1.3)
        img = img.filter(ImageFilter.SHARPEN)
        text = pytesseract.image_to_string(img)
        return text.strip() or "[No text detected]"
    except Exception as e:
        return f"[OCR error: {str(e)}]"

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
            try:
                images = convert_from_path(path)
                return "\n".join(enhance_and_ocr(img) for img in images)
            except Exception as e:
                return f"[PDF OCR error: {str(e)}]"
        return "[PDF OCR unavailable]"

# === OpenRouter / OpenAI Integration (for text model calls) ===
try:
    import openai
    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False

def call_openrouter_model(prompt: str, model: str, api_key: str) -> str:
    """
    Tries OpenAI SDK configured to use OpenRouter. If SDK not available or fails,
    returns a helpful error message. We don't change your model names here.
    """
    if not api_key:
        return "[❌ OpenRouter API key not provided]"
    if not HAS_OPENAI_SDK:
        return "[⚠️ OpenAI SDK missing. Run: pip install openai==0.28.0]"

    try:
        # Use OpenAI SDK but point it to the OpenRouter base if required.
        openai.api_key = api_key
        openai.api_base = "https://openrouter.ai/api/v1"
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a structured business & cost analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        # return the exception message for logging/inspection
        return f"[❌ Model call failed: {str(e)}]"

# === Pictator image generation — SAFE approach ===
# Primary: Hugging Face Router (requires HF_TOKEN). Secondary: try existing SDKs where possible.
def hf_router_generate_image(hf_token: str, model_repo: str, prompt: str, width=1024, height=1024, steps=30, guidance=3.5):
    if not hf_token:
        return {"type": "error", "data": "[HF token not provided]"}
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
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except Exception as e:
        return {"type": "error", "data": f"[HF Router request failed: {str(e)}]"}

    # No content
    if not resp.content or resp.text.strip() == "":
        return {"type": "error", "data": "HF Router returned empty response."}

    # If binary image
    ctype = resp.headers.get("content-type", "")
    if "image" in ctype:
        try:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return {"type": "image", "data": img}
        except Exception as e:
            return {"type": "error", "data": f"[Failed to decode image bytes: {str(e)}]"}

    # Try JSON responses with base64 images
    try:
        data = resp.json()
    except Exception:
        # Show raw for debugging
        return {"type": "error", "data": f"[HF Router non-JSON response]: {resp.text[:200]}"}

    # JSON formats (several variants)
    try:
        if isinstance(data, dict) and "generated_image" in data:
            img_bytes = base64.b64decode(data["generated_image"])
            return {"type": "image", "data": Image.open(io.BytesIO(img_bytes)).convert("RGB")}
        if isinstance(data, dict) and "image" in data:
            img_bytes = base64.b64decode(data["image"])
            return {"type": "image", "data": Image.open(io.BytesIO(img_bytes)).convert("RGB")}
        if isinstance(data, dict) and "images" in data and len(data["images"])>0:
            img_bytes = base64.b64decode(data["images"][0])
            return {"type": "image", "data": Image.open(io.BytesIO(img_bytes)).convert("RGB")}
        if isinstance(data, list) and len(data)>0:
            first = data[0]
            if isinstance(first, dict):
                # common field names: blob, generated_image
                for key in ("blob","generated_image","image"):
                    if key in first:
                        img_bytes = base64.b64decode(first[key])
                        return {"type": "image", "data": Image.open(io.BytesIO(img_bytes)).convert("RGB")}
    except Exception as e:
        return {"type": "error", "data": f"[Error parsing HF JSON -> {str(e)}]"}

    return {"type": "error", "data": f"[Unsupported HF response: {json.dumps(data)[:400]}]"}

# === generate_pictator_image: unified entry used by Tab 4 ===
def generate_pictator_image(prompt: str, engine_choice: str, openrouter_key: str, hf_token: str, width=1024, height=1024, steps=30, guidance=3.5):
    """
    engine_choice can be:
      - 'openrouter:<model>'  (calls call_openrouter_model for text-to-image if supported)
      - 'hf:<model_repo>'     (uses Hugging Face Router to generate image)
      - 'gpt' / 'claude' / 'gemini' triggers best-effort branches if SDKs available
    """
    engine_choice = engine_choice.strip()
    # HF Router branch
    if engine_choice.startswith("hf:"):
        model_repo = engine_choice.split("hf:",1)[1]
        return hf_router_generate_image(hf_token, model_repo, prompt, width, height, steps, guidance)

    # OpenRouter / OpenAI branch (best effort for text generation or image if SDK supports)
    if engine_choice.startswith("openrouter:"):
        model = engine_choice.split("openrouter:",1)[1]
        # We will call openrouter for a text response that might include an image URL, or error
        text_resp = call_openrouter_model(prompt, model, openrouter_key)
        return {"type":"text","data":text_resp}

    # Specialized SDKs (gpt/claude/gemini) — attempt existing code paths with safe fallbacks:
    eld = engine_choice.lower()
    if "gpt" in eld:
        # try to use OpenAI SDK with provided openrouter_key (if present)
        if not openrouter_key:
            return {"type":"error","data":"OpenRouter key not provided for GPT branch."}
        try:
            if not HAS_OPENAI_SDK:
                return {"type":"error","data":"OpenAI SDK missing; cannot use GPT-vision branch."}
            # This branch mostly supports text or vision SDKs; many environments don't allow direct image generation this way.
            openai.api_key = openrouter_key
            openai.api_base = "https://openrouter.ai/api/v1"
            # Attempt chat completion and check for image url in response (best-effort)
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=1500
            )
            txt = completion["choices"][0]["message"]["content"]
            return {"type":"text","data":txt}
        except Exception as e:
            return {"type":"error","data":f"[GPT branch error: {str(e)}]"}

    if "claude" in eld:
        return {"type":"error","data":"Claude SDK branch not implemented in this environment."}

    if "gemini" in eld:
        return {"type":"error","data":"Gemini SDK branch not implemented in this environment."}

    return {"type":"error","data":"Unsupported pictator engine choice."}


# === Streamlit UI ===
st.set_page_config(page_title="Harmony Costulator + Pictator Analyzer (v1.5.4)", layout="wide")
st.title("⚡ Harmony Costulator + Pictator Analyzer (v1.5.4)")

# --- API keys UI
st.sidebar.header("API Keys / Secrets")
openrouter_key_input = st.sidebar.text_input("OpenRouter Key (for analysis)", type="password")
# Try to read HF token from app secrets first; allow override from sidebar
hf_token_secret = st.secrets.get("HF_TOKEN") if "HF_TOKEN" in st.secrets else None
hf_token_input = st.sidebar.text_input("HuggingFace Token (override)", type="password", value=hf_token_secret or "")

api_key = openrouter_key_input  # legacy variable name used below for call_openrouter_model

tabs = st.tabs(["🧠 Pictator Analyzer", "📊 Costulator (Profitability)", "📈 Costulator Generator", "🎨 Pictator Creator"])

# === TAB 1: Pictator Analyzer ===
with tabs[0]:
    st.subheader("Upload Drawing / Design Image or PDF")
    file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])
    custom_prompt = st.text_area(
        "Custom Prompt (editable)",
        "Analyze this engineering drawing for materials, machining process, tooling setup, optimization, and improvements."
    )

    extracted_text = ""
    if file:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            tmp = f"temp_{int(time.time())}_{file.name}"
            with open(tmp, "wb") as f:
                f.write(file.read())
            with st.spinner("Extracting text from PDF..."):
                extracted_text = ocr_pdf(tmp)
            try:
                os.remove(tmp)
            except Exception:
                pass
        else:
            img = Image.open(file)
            st.image(img, caption="Uploaded Drawing", use_container_width=True)
            with st.spinner("Performing OCR on image..."):
                extracted_text = enhance_and_ocr(img)

        st.text_area("📜 Extracted Text", extracted_text, height=180)

        if st.button("🔍 Run Pictator Analysis"):
            st.info("Running Pictator Analyzer...")
            # summary using OpenRouter (best-effort)
            summary = call_openrouter_model(f"Summarize this drawing:\n{extracted_text}", "meta-llama/llama-4-scout:free", api_key)
            st.markdown("### 📘 Drawing Summary")
            st.write(summary)

            analysis = call_openrouter_model(f"{custom_prompt}\n\nDrawing text:\n{extracted_text}", "meta-llama/llama-3.3-70b-instruct:free", api_key)
            st.subheader("🧩 Pictator AI Engineering Insights")
            st.write(analysis)
            st.download_button("⬇️ Download Pictator Analysis", data=analysis, file_name="pictator_analysis.txt")

# === TAB 2: Costulator (Profitability) ===
with tabs[1]:
    st.subheader("Upload Costing Sheet or Report")
    cost_file = st.file_uploader("Upload costing image, CSV, or PDF", type=["csv", "jpg", "jpeg", "png", "pdf"])
    cost_prompt = st.text_area(
        "Custom Prompt",
        "Analyze this costing sheet for profitability and generate a 3–9 month cost optimization plan."
    )

    text_data = ""
    df = None
    if cost_file:
        ext = cost_file.name.split(".")[-1].lower()
        if ext == "csv":
            try:
                df = pd.read_csv(cost_file)
                st.dataframe(df)
                text_data = df.to_string(index=False)
            except Exception as e:
                st.error(f"[CSV parse error: {str(e)}]")
                text_data = ""
        elif ext == "pdf":
            tmp = f"temp_{int(time.time())}_{cost_file.name}"
            with open(tmp, "wb") as f:
                f.write(cost_file.read())
            text_data = ocr_pdf(tmp)
            try:
                os.remove(tmp)
            except Exception:
                pass
        else:
            img = Image.open(cost_file)
            st.image(img, caption="Uploaded Costing Image", use_container_width=True)
            text_data = enhance_and_ocr(img)

        st.text_area("🧾 Extracted Cost Data", text_data, height=200)

        if st.button("💰 Run Costulator Analysis"):
            st.info("Running Costulator Analysis...")

            summary = call_openrouter_model(f"Summarize costing:\n{text_data}", "deepseek/deepseek-r1-distill-llama-70b:free", api_key)
            st.markdown("### 📊 Cost Summary")
            st.write(summary)

            analysis_prompt = f"""
{cost_prompt}

Costing Data:
{text_data}

Return structured data:
- Cost components
- Optimization & savings strategies
- Cost reduction plan (0–3, 3–6, 6–9, 9–12 months)
"""
            analysis = call_openrouter_model(analysis_prompt, "openai/gpt-oss-20b:free", api_key)
            st.subheader("💡 Costulator AI Recommendations")
            st.write(analysis)

            st.session_state["costulator_analysis"] = analysis
            st.session_state["costulator_df"] = df if df is not None else None
            st.download_button("⬇️ Download Costulator Report", data=analysis, file_name="costulator_analysis.txt")

# === TAB 3: Costulator Generator (Auto Forecast Table) ===
with tabs[2]:
    st.subheader("📈 Generate Forecasted Cost Sheet (Based on Costulator Analysis)")
    if "costulator_df" not in st.session_state or st.session_state["costulator_df"] is None:
        st.warning("Please upload and analyze a costing sheet first in Tab 2.")
    else:
        df = st.session_state["costulator_df"]
        analysis_text = st.session_state.get("costulator_analysis", "")

        st.info("Generating revised costing table with projections (0–3, 3–6, 6–9, 9–12 months)...")

        # Build prompt only if df exists
        table_prompt = f"""
From the below cost analysis and table, generate a revised CSV maintaining same columns and structure.
Add four new columns: cost_0_3m, cost_3_6m, cost_6_9m, cost_9_12m.
Each column should represent reduced costs based on forecasted savings mentioned or inferred from analysis.

Original Cost Table:
{df.to_csv(index=False)}

AI Analysis Context:
{analysis_text}

Return CSV table.
"""
        revised_csv = call_openrouter_model(table_prompt, "mistralai/mistral-7b-instruct:free", api_key)
        st.subheader("📊 Revised Forecasted Cost Sheet")
        st.code(revised_csv, language="csv")
        st.download_button("⬇️ Download Revised Costing Sheet (Excel)",
                           data=revised_csv, file_name="revised_costing_forecast.csv", mime="text/csv")

# === TAB 4: Pictator Creator ===
with tabs[3]:
    st.subheader("Create Engineering Drawing from Text")

    # Provide choices: hf:repo or openrouter:MODEL or SDK names (user-friendly)
    pictator_choices = {
        "HF: FLUX.1-dev (lineart/mech)": "hf:black-forest-labs/FLUX.1-dev",
        "HF: SDXL (2D blueprint)": "hf:stabilityai/stable-diffusion-xl-base-1.0",
        "HF: RealisticVision (3D Render)": "stabilityai/stable-diffusion-3-medium-diffusers",       
        "OpenRouter: meta-llama/llama-4-scout (text analysis)": "openrouter:meta-llama/llama-4-scout:free"
    }
    model_name = st.selectbox("Select Pictator Engine", list(pictator_choices.keys()))
    engine_choice = pictator_choices[model_name]

    drawing_prompt = st.text_area(
        "Drawing Prompt (editable)",
        """Create a detailed mechanical engineering drawing of a CNC component showing diameter, hole distances,
tolerances, finishing, labels, and scaling specifications. Provide clean lineart suitable for CNC cutting."""
    )

    colA, colB = st.columns(2)
    with colA:
        pict_width = st.number_input("Width", 256, 1536, 1024)
    with colB:
        pict_height = st.number_input("Height", 256, 1536, 1024)

    pict_steps = st.slider("Inference Steps", 10, 80, 30)
    pict_guidance = st.slider("Guidance Scale", 1.0, 12.0, 3.5)

    if st.button("🎨 Generate Pictator Drawing"):
        with st.spinner(f"Generating drawing using {model_name}..."):
            # engine_choice might be 'hf:xxx' or 'openrouter:...'
            hf_token_to_use = hf_token_input or hf_token_secret
            result = generate_pictator_image(
                drawing_prompt,
                engine_choice,
                openrouter_key_input,
                hf_token_to_use,
                width=pict_width,
                height=pict_height,
                steps=pict_steps,
                guidance=pict_guidance
            )

        if result.get("type") == "image":
            img = result["data"]
            st.image(img, caption=f"Generated by {model_name}", use_container_width=True)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.session_state["pictator_image"] = buf.getvalue()
            st.download_button("⬇️ Download Drawing (PNG)",
                               data=st.session_state["pictator_image"],
                               file_name="pictator_generated.png",
                               mime="image/png")
        elif result.get("type") == "text":
            st.info("Text result:")
            st.write(result["data"])
        else:
            st.error(result.get("data", "Unknown error"))

st.markdown("---")
st.caption("© 2025 Harmony Strategy Partner — Costulator + Pictator Suite (Gemini + GPT + DeepSeek + OCR)")
