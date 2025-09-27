import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import sys
import glob
import os


def main():
    if len(sys.argv) < 4:
        print("Usage: python caption.py <file_glob> <mode> <length>")
        sys.exit(1)

    # Simple VQA
    image_paths = glob.glob(sys.argv[1])
    mode = sys.argv[2]

    if mode not in ["stdout", "txt"]:
        print("Invalid mode. Must be one of: stdout, txt")
        return

    length = sys.argv[3]

    if length not in ["short", "normal", "long"]:
        print("Invalid length. Must be one of: short, normal, long")
        return

    moondream = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )

    moondream.compile()

    for image_path in image_paths:
        image = Image.open(image_path)
        result = moondream.caption(image=image, length=length)
        print(f"{image_path}: {result['caption']}")

        if mode == "txt":
            # replace original extension
            filename = os.path.splitext(image_path)[0] + ".txt"
            with open(filename, "w") as f:
                filename = os.path.splitext(image_path)[0] + ".txt"
                with open(filename, "w") as f:
                    f.write(result["caption"])
                    print(f"Caption saved to {filename}")


if __name__ == "__main__":
    main()
