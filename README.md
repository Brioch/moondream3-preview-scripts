# Moondream 3 scripts

A collection of scripts to run inference on the [Moondream 3 (Preview)](https://huggingface.co/moondream/moondream3-preview) vision language model.

## Requirements

- Python (tested on 3.13)

## Installation

- Clone the repository.
- Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### query.py

This script allows you to query the model with a question about an image.

```bash
python query.py <image_path> <question>
```

Parameters :

- `<image_path>`: The path to the image you want to query.
- `<question>`: The question you want to ask about the image.

### chat.py

This script allows you to ask several questions about an image without having to load the model each time.

```bash
python chat.py <image_path>
```

Parameters :

- `<image_path>`: The path to the image you want to query.

### caption.py

This script allows you to generate captions for one or multiple images.
Captions will be output in the console or in `.txt` files next to their corresponding images.

```bash
python caption.py <file_glob> <mode> <length>
```

Parameters :

- `<file_glob>`: A glob pattern to match the images you want to caption.
- `<mode>`: The mode you want to use. Can be one of: `stdout`, `txt`.
    - In case of `stdout`, the captions will be output in the console.
    - In case of `txt`, the captions will be saved in `.txt` files next to their corresponding images.
- `<length>`: The length of the caption. Can be one of: `short`, `normal`, `long`.

### moondream.py

This script allows you to use the different modes supported by the model.

```bash
python moondream.py <image_path> <mode> <argument>
```

Parameters :

- `<image_path>`: The path to the image you want to query.
- `<mode>`: The mode you want to use. Can be one of: `question`, `caption`, `point`, `detect`.
- `<argument>`: The argument to pass to the mode.
    - In case of `question`, it is the question you want to ask about the image.
    - In case of `caption`, it is the length of the caption. Can be one of: `short`, `normal`, `long`.
    - In case of `point`, it is the part of the image you want the coordinates of.
    - In case of `detect`, it is the object you want to detect in the image.

## Disclaimer

This project is not affiliated with the MoonDream project. It is a simple wrapper around the MoonDream model.

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for more information.