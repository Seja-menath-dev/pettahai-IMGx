from gradio_client import Client
import gradio as gr

# Hugging Face API Key (if required for the client)
API_KEY = "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"

# Initialize the Gradio Client for DALLE-4K
client = Client("mukaist/DALLE-4K", hf_token=API_KEY)

# Function to generate the image using the DALLE-4K API
def generate_dalle4k_image(prompt, negative_prompt, use_negative_prompt, style, seed, width, height, guidance_scale, randomize_seed):
    result = client.predict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_negative_prompt=use_negative_prompt,
        style=style,
        seed=seed,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        randomize_seed=randomize_seed,
        api_name="/run"
    )
    
    # Return the generated image and seed
    return result[0][0]['image'], result[1]

# Gradio Interface with all the required inputs
def run_gradio():
    interface = gr.Interface(
        fn=generate_dalle4k_image,
        inputs=[
            gr.Textbox(placeholder="Enter your prompt here", label="Prompt"),
            gr.Textbox(placeholder="Enter negative prompt (optional)", label="Negative Prompt", value="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy..."),
            gr.Checkbox(label="Use Negative Prompt", value=True),
            gr.Radio(choices=['3840 x 2160', '2560 x 1440', 'Photo', 'Cinematic', 'Anime', '3D Model', '(No style)'], label="Image Style", value="3840 x 2160"),
            gr.Slider(minimum=0, maximum=1000, step=1, label="Seed", value=0),
            gr.Slider(minimum=512, maximum=2048, step=64, label="Width", value=1024),
            gr.Slider(minimum=512, maximum=2048, step=64, label="Height", value=1024),
            gr.Slider(minimum=1, maximum=20, step=1, label="Guidance Scale", value=6),
            gr.Checkbox(label="Randomize Seed", value=True),
        ],
        outputs=[
            gr.Image(label="Generated Image"),
            gr.Textbox(label="Seed Used")
        ],
        title="Pettahai-IMGx Image Generator",
        description="Generate high-quality images using the Pettahai-IMGx model from text prompts.",
    )

    # Launch Gradio interface on port 5000
    interface.launch(server_port=5000, share=True)

run_gradio()

