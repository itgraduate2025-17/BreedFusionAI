

import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
import warnings
import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline

warnings.filterwarnings("ignore")

# ============================================================
# 📌 LOAD CSV
# ============================================================
CSV_PATH = "updated animal_full_dataset_with_nature.csv"  # upload this to HF Space
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    df = None
    print("⚠️ CSV file not found or failed to load:", e)

# ============================================================
# 📌 LOAD TEXT MODEL
# ============================================================
try:
    text_pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
except Exception as e:
    print("Text model error:", e)
    text_pipe = None

# ============================================================
# 📌 LOAD IMAGE MODEL
# ============================================================
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    image_pipe = image_pipe.to(device)
    image_pipe.set_progress_bar_config(disable=True)
except Exception as e:
    print("Image model error:", e)
    image_pipe = None

# ============================================================
# 📌 HELPER FUNCTIONS
# ============================================================
def get_image_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=3)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = img.resize((200, 200))
        return img
    except:
        # Return placeholder if image fails
        placeholder = Image.new("RGB", (200,200), (100,100,100))
        return placeholder

def generate_hybrid(parent1_breed, parent2_breed):
    if not parent1_breed or not parent2_breed:
        return "⚠️ Please select both breeds!", None, None
    try:
        p1 = df[df["Breed"] == parent1_breed].iloc[0]
        p2 = df[df["Breed"] == parent2_breed].iloc[0]
    except:
        return "❌ Breed not found in dataset!", None, None

    # Parent Images side by side
    img1 = get_image_from_url(p1["Image_URL"])
    img2 = get_image_from_url(p2["Image_URL"])
    combined = Image.new("RGB", (410, 200), (0,0,0))
    combined.paste(img1, (0,0))
    combined.paste(img2, (210,0))

    # Hybrid text prompt
    prompt = f"""
<|system|>You are a geneticist AI.<|user|>
Combine these animals:
1. {p1['Breed']} ({p1['Nature']}, {p1['Lifespan_Years']}y, Danger: {p1['Danger_Level']})
2. {p2['Breed']} ({p2['Nature']}, {p2['Lifespan_Years']}y, Danger: {p2['Danger_Level']})
Output Format:
Hybrid Name: [Name]
Lifespan: [Number] years
Danger Level: [Level]
Nature: [Adjective]
Description: [Short visual description]
<|assistant|>
"""

    # Generate hybrid text
    if text_pipe:
        raw = text_pipe(prompt, max_new_tokens=150, do_sample=True)[0]["generated_text"]
        final_text = raw.split("<|assistant|>")[-1].strip()
    else:
        final_text = "⚠️ Text model unavailable."

    # Extract hybrid name & description
    hybrid_name, desc = "Hybrid Animal", "A unique hybrid creature."
    for line in final_text.split("\n"):
        if "Hybrid Name:" in line: hybrid_name = line.split(":")[-1].strip()
        if "Description:" in line: desc = line.split(":")[-1].strip()

    # Generate hybrid image
    if image_pipe:
        img_prompt = f"{desc}, hybrid of {p1['Breed']} and {p2['Breed']}, ultra detailed, photorealistic"
        hybrid_img = image_pipe(img_prompt, num_inference_steps=20).images[0]
    else:
        hybrid_img = None

    return final_text, combined, hybrid_img

# ============================================================
# 📌 DROPDOWN SETUP
# ============================================================
if df is not None:
    animal_types = sorted(df["Animal"].unique().tolist())
else:
    animal_types = []

def get_breeds(animal):
    if df is None:
        return []
    return sorted(df[df["Animal"] == animal]["Breed"].unique().tolist())

def update_breed_dropdown(animal):
    breeds = get_breeds(animal)
    default = breeds[0] if breeds else None
    return gr.update(choices=breeds, value=default)

# ============================================================
# 📌 GRADIO UI
# ============================================================
with gr.Blocks(title="BreedFusion AI") as app:

    gr.Markdown("## 🧬 **BreedFusion AI — Hybrid Generator**")

    with gr.Row():
        animal1 = gr.Dropdown(animal_types, label="Parent 1 Animal")
        breed1 = gr.Dropdown([], label="Parent 1 Breed")

    with gr.Row():
        animal2 = gr.Dropdown(animal_types, label="Parent 2 Animal")
        breed2 = gr.Dropdown([], label="Parent 2 Breed")

    # Update breed dropdowns automatically
    animal1.change(update_breed_dropdown, inputs=animal1, outputs=breed1)
    animal2.change(update_breed_dropdown, inputs=animal2, outputs=breed2)

    btn = gr.Button("Generate Hybrid", variant="primary")

    out_text = gr.Textbox(label="Hybrid Information", lines=8)
    out_parents = gr.Image(label="Parent Images")
    out_hybrid = gr.Image(label="Hybrid Result")

    btn.click(
        fn=generate_hybrid,
        inputs=[breed1, breed2],
        outputs=[out_text, out_parents, out_hybrid],
        queue=True
    )

app.launch()
