from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import field, parse_known_args
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F

from flask import Flask, flash, request, redirect, json
from werkzeug.utils import secure_filename

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels, combined_names


@dataclass
class ScriptOptions:
    image_file: BytesIO
    gen_threshold: float = 0.35
    char_threshold: float = 0.75

model_name = 'vit'
repo_id = MODEL_REPO_MAP.get(model_name)

print(f"Loading model '{model_name}' from '{repo_id}'...")
model: nn.Module = timm.create_model("hf-hub:" + repo_id).eval()
state_dict = timm.models.load_state_dict_from_hf(repo_id)
model.load_state_dict(state_dict)

print("Loading tag list...")
labels: LabelData = load_labels_hf(repo_id=repo_id)

print("Creating data transform...")
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

def main(opts: ScriptOptions):

    print("Loading image and preprocessing...")
    # get image
    img_input: Image.Image = Image.open(opts.image_file)
    # ensure image is RGB
    img_input = pil_ensure_rgb(img_input)
    # pad to square with white background
    img_input = pil_pad_square(img_input)
    # run the model's input transform to convert to tensor and rescale
    inputs: Tensor = transform(img_input).unsqueeze(0)
    # NCHW image RGB to BGR
    inputs = inputs[:, [2, 1, 0]]

    print("Running inference...")
    with torch.inference_mode():
        model_1 = model
        # move model to GPU, if available
        if torch_device.type != "cpu":
            model_1 = model_1.to(torch_device)
            inputs = inputs.to(torch_device)
        # run the model
        outputs = model.forward(inputs)
        # apply the final activation function (timm doesn't support doing this internally)
        outputs = F.sigmoid(outputs)
        # move inputs, outputs, and model back to to cpu if we were on GPU
        if torch_device.type != "cpu":
            inputs = inputs.to("cpu")
            outputs = outputs.to("cpu")
            model_1 = model_1.to("cpu")

    print("Processing results...")
    caption, taglist, ratings, character, general, caption_split = get_tags(
        probs=outputs.squeeze(0),
        labels=labels,
        gen_threshold=opts.gen_threshold,
        char_threshold=opts.char_threshold,
    )

    print("--------")
    print(f"Caption: {caption}")
    print("--------")
    print(f"Tags: {taglist}")

    print("--------")
    print("Ratings:")
    for k, v in ratings.items():
        print(f"  {k}: {v:.3f}")

    print("--------")
    print(f"Character tags (threshold={opts.char_threshold}):")
    for k, v in character.items():
        print(f"  {k}: {v:.3f}")

    print("--------")
    print(f"General tags (threshold={opts.gen_threshold}):")
    for k, v in general.items():
        print(f"  {k}: {v:.3f}")

    print("Done!")

    return [f'{x}' for x in caption_split]


def run_server(threshold: float, host: str, port: int, cpu: bool, auth_key: Optional[str]):
    app = Flask("hydrus-dd lookup server")
    #ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'webp'])

    #def allowed_file(filename: str | None):
    #    return filename is not None and '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if auth_key:
                if request.headers.get('authorization') != f'Bearer {auth_key}':
                    response = app.response_class(
                        response=json.dumps({'error': 'invalid auth key'}),
                        status=401,
                        mimetype='application/json'
                    )
                    return response

            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            if file:
                _ = secure_filename(file.filename)
                image_path = BytesIO(file.read())
                results = main(ScriptOptions(image_path))
                deepdanbooru_response = json.dumps(results),
                response = app.response_class(
                    response=deepdanbooru_response,
                    status=200,
                    mimetype='application/json'
                )
                return response
            else:
                response = app.response_class(
                    response=json.dumps({'error': 'invalid file'}),
                    status=400,
                    mimetype='application/json'
                )

        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <p><input type=file name=file>
             <input type=submit value=Upload>
        </form>
        '''

    app.run(host=host, port=port)

if __name__ == '__main__':
    run_server(float(os.environ.get('THRESHOLD') or 0.5), '127.0.0.1', 12152, True, os.environ.get('AUTH_KEY'))
