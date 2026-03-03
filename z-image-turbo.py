import replicate
import os

# We'll set this in Colab, but this function runs the model
def generate_image(prompt, api_token):
    os.environ["r8_0yOK7yilUfJYkKn6JBSPltXagjFiqUZ48vY0S"] = api_token
    
    # Using 'flux-schnell' because it's fast (great for testing)
    output = replicate.run(
        "prunaai/z-image-turbo",
        input={
            "prompt": prompt,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80
        }
    )
    # Replicate returns a list of URLs
    return output[0]
