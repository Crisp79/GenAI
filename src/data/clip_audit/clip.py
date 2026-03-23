import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


def run_clip_audit(csv_path, image_dir, output_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    text_prompts = [
        "a photo of a face without glasses",
        "a photo of a face wearing glasses"
    ]

    df = pd.read_csv(csv_path)

    predictions = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(image_dir, row["image_name"])

            try:
                image = Image.open(img_path).convert("RGB")
            except:
                predictions.append(-1)
                continue

            inputs = processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

            pred = torch.argmax(probs, dim=1).item()
            predictions.append(pred)

    df["clip_prediction"] = predictions

    df.to_csv(output_path, index=False)
    return df