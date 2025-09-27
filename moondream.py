import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import sys


def main():
    if len(sys.argv) != 4:
        print("Usage: python moondream.py <image_path> <mode> <argument>")
        sys.exit(1)

    image_path = sys.argv[1]
    mode = sys.argv[2]  # can be one of: question, caption, point, detect

    if mode not in ["question", "caption", "point", "detect"]:
        print("Invalid mode. Must be one of: question, caption, point, detect")
        return

    image = Image.open(image_path)
    print(f"Image '{image_path}' loaded.")

    moondream = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    moondream.compile()
    argument = sys.argv[3]

    if mode == "question":
        result = moondream.query(image=image, question=argument)
        print(result["answer"])

    elif mode == "caption":
        if argument not in ["short", "long", "normal"]:
            print("Usage: python moondream.py <image_path> caption <length>")
            return

        result = moondream.caption(image, length=argument)
        print(result["caption"])

    elif mode == "point":
        result = moondream.point(image, argument)

        for i, point in enumerate(result["points"]):
            print(f"Point {i+1}: x={point['x']:.3f}, y={point['y']:.3f}")

    elif mode == "detect":
        result = moondream.detect(image, argument)

        for i, obj in enumerate(result["objects"]):
            print(
                f"Object {i+1}: "
                f"x_min={obj['x_min']:.3f}, y_min={obj['y_min']:.3f}, "
                f"x_max={obj['x_max']:.3f}, y_max={obj['y_max']:.3f}"
            )

    else:
        print("Invalid mode. Use 'chat', 'caption', 'point', or 'detect'.")


if __name__ == "__main__":
    main()
