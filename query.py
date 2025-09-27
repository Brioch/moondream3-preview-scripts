import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import sys

moondream = AutoModelForCausalLM.from_pretrained(
    "moondream/moondream3-preview",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": "cuda"},
)
moondream.compile()

if len(sys.argv) < 3:
    print("Usage: python query.py <image_path> <question>")
    sys.exit(1)

# Simple VQA
image_path = sys.argv[1]
image = Image.open(image_path)
question = sys.argv[2]
result = moondream.query(image=image, question=question)
print(result["answer"])
