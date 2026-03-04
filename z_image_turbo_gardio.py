import os
import random
import requests
import gradio as gr
import replicate
from PIL import Image
from io import BytesIO

def get_dimensions(ratio_str, max_side=1536):
    """Calculates width and height based on ratio string to hit max resolution."""
    ratios = {
        "1:1": (1, 1), "16:9": (16, 9), 
        "3:2": (3, 2), "2:3": (2, 3), "3:4": (3, 4), "4:3": (4, 3), 
        "9:16": (9, 16)
    }
    rw, rh = ratios.get(ratio_str, (1, 1))
    
    if rw > rh:
        w = max_side
        h = int((max_side * rh) / rw)
    else:
        h = max_side
        w = int((max_side * rw) / rh)
    
    # Ensure dimensions are multiples of 8 for the model
    return (w // 8) * 8, (h // 8) * 8

def generate_image(prompt, aspect_ratio, num_inference_steps, seed, randomize_seed):
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise gr.Error("REPLICATE_API_TOKEN not found.")

    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    
    # Calculate target dimensions for 2048px resolution
    width, height = get_dimensions(aspect_ratio, max_side=2048)
    
    try:
        # Using Flux-Schnell with explicit dimensions and format
        output = replicate.run(
            "prunaai/z-image-turbo",
            input={
                "prompt": prompt,
                "seed": int(seed),
                "num_inference_steps": int(num_inference_steps),
                "width": width,
                "height": height,
                "output_format": output_format,
                "output_quality": int(output_quality)
            }
        )
        
        # Handle Output
        if isinstance(output, list) and len(output) > 0:
            image_url = output[0]
        elif hasattr(output, 'url'):
            image_url = output.url
        else:
            # Fallback for direct binary stream
            img = Image.open(BytesIO(output.read()))
            return img, seed

        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image, seed

    except Exception as e:
        raise gr.Error(f"Replicate Error: {str(e)}")

# Custom theme
custom_theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="amber",
    neutral_hue="slate"
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("# Z-Image-Turbo (via Replicate API)\nGenerate realistic images with 'prunaai/z-image-turbo'")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=4)
            aspect_ratio = gr.Dropdown(
                choices=["1:1", "16:9", "3:2", "2:3", "3:4", "4:3", "9:16"],
                value="3:4", label="Aspect Ratio"
            )
            steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
            output_quality = gr.Slider(1,100, value=80, step=1, label="Output quality")
            output_format = gr.Dropdown(
                choices=["jpg", "png", "webp"], value="jpg", label="Output format")
            
            with gr.Row():
                randomize = gr.Checkbox(label="Random Seed", value=True)
                seed = gr.Number(label="Seed", value=42, visible=False)
            
            btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            # Use 'png' format in the Image component to ensure Gradio displays it correctly
            out_img = gr.Image(label="Generated image:", type="pil", format=output_format, show_download_button=True)
            out_seed = gr.Number(label="Used Seed", interactive=False)

    randomize.change(lambda r: gr.update(visible=not r), randomize, seed)
    btn.click(generate_image, [prompt, aspect_ratio, steps, seed, randomize], [out_img, out_seed])

if __name__ == "__main__":
    demo.launch(share=True)
