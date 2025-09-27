import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python chat.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path)
    print(f"Image '{image_path}' loaded.")

    moondream = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    moondream.compile()

    encoded = moondream.encode_image(image)

    while True:
        print("\nEnter your question (or 'quit' to exit):")

        q = input().strip()
        if q.lower() == "quit":
            print("Exiting...")
            break

        print("Processing...")

        try:
            res = moondream.query(image=encoded, question=q, stream=True)

            for chunk in res["answer"]:
                print(chunk, end="", flush=True)

        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    main()
