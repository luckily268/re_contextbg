import base64, io
from PIL import Image
from openai import OpenAI

client = OpenAI(
    api_key="sk-0pOwV2F4WDI6cwA5jywVJd7T4FPYuIzbKzMllZQCYyPVxtDu",
    base_url="https://ergouzi.life/v1"
)

img = Image.open("/home/dachuang/ycx/dataset/samples/CAM_BACK/n003-2018-01-03-12-03-23+0800__CAM_BACK__1514952316316487.jpg")
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=85)
b64 = base64.b64encode(buf.getvalue()).decode()

print("Sending image to model...")
resp = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "Describe the scene in this image briefly."}
    ]}],
    max_tokens=200
)
print(resp.choices[0].message.content)
