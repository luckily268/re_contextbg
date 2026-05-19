# Image processing utilities for multimodal backdoor attacks
import base64
import io
import os
import random

from PIL import Image, ImageDraw, ImageFont


def add_red_trigger(image, position="top_right"):
    """Add a semi-transparent red rectangular trigger to the image.
    This simulates a physical visual marker placed in the environment."""
    img = image.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = 160, 90
    pad = 15
    positions = {
        "top_right": (img.width - w - pad, pad, img.width - pad, h + pad),
        "top_left": (pad, pad, w + pad, h + pad),
        "center": ((img.width - w) // 2, (img.height - h) // 2,
                   (img.width + w) // 2, (img.height + h) // 2),
    }
    x1, y1, x2, y2 = positions.get(position, positions["top_right"])
    draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 180))
    draw.rectangle([x1, y1, x2, y2], outline=(200, 0, 0, 255), width=3)

    result = Image.alpha_composite(img, overlay)
    return result.convert('RGB')


def image_to_base64(image, fmt='JPEG'):
    """Convert PIL Image to base64 string for API calls."""
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def load_images(directory, num=None, seed=42):
    """Load images from directory. Returns list of (filename, PIL.Image)."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted(f for f in os.listdir(directory) if f.lower().endswith(exts))
    if num and num < len(files):
        random.seed(seed)
        files = sorted(random.sample(files, num))
    return [(f, Image.open(os.path.join(directory, f))) for f in files]


def build_image_url(b64_str, fmt='jpeg'):
    """Build a data URL for the OpenAI-compatible multimodal API."""
    return f"data:image/{fmt};base64,{b64_str}"
