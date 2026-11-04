"""Gradio frontend for FLUX.1-dev served by vLLM-Omni (OpenAI-compatible /v1/images/generations)."""

import base64
import io
import os
import random

import gradio as gr
import requests
from PIL import Image

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8091").rstrip("/")
MODEL = os.environ.get("MODEL", "black-forest-labs/FLUX.1-dev")
TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "600"))
MAX_SEED = 2**31 - 1


def generate(prompt, width, height, steps, guidance, seed, randomize_seed, num_images):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is empty.")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "n": int(num_images),
        # FLUX works on 16px latent patches; snap to a multiple of 16
        "size": f"{int(width) // 16 * 16}x{int(height) // 16 * 16}",
        "num_inference_steps": int(steps),
        # FLUX.1-dev is guidance-distilled: this is the embedded guidance, not real CFG
        "guidance_scale": float(guidance),
        "seed": int(seed),
        "response_format": "b64_json",
    }
    try:
        r = requests.post(f"{VLLM_URL}/v1/images/generations", json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise gr.Error(f"Cannot reach vLLM backend at {VLLM_URL}: {e}")
    if r.status_code != 200:
        raise gr.Error(f"Backend error {r.status_code}: {r.text[:500]}")

    images = [
        Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))
        for item in r.json()["data"]
    ]
    return images, seed


with gr.Blocks(title="FLUX.1-dev") as demo:
    gr.Markdown("# FLUX.1-dev\nServed by vLLM-Omni")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=4, placeholder="A cat holding a sign that says hello world")
            run = gr.Button("Generate", variant="primary")
            with gr.Accordion("Settings", open=True):
                with gr.Row():
                    width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
                steps = gr.Slider(1, 50, value=28, step=1, label="Inference steps")
                guidance = gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Guidance")
                num_images = gr.Slider(1, 4, value=1, step=1, label="Images")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed", minimum=0, maximum=MAX_SEED)
                    randomize_seed = gr.Checkbox(value=True, label="Randomize seed")
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Result", columns=2, height=640, format="png")

    inputs = [prompt, width, height, steps, guidance, seed, randomize_seed, num_images]
    run.click(generate, inputs, [gallery, seed])
    prompt.submit(generate, inputs, [gallery, seed])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
    )
