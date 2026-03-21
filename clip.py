#%%
import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm.notebook import tqdm

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

# Load the model and the processor (which handles resizing and text tokenization)
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# 1. Define your exact text prompts
text_prompts = ["a photo of a face without glasses", "a photo of a face wearing glasses"]

# Load your dataframe
df = pd.read_csv('data/test.csv')
df['id'] = df['id'].astype(int).astype(str)
df['id'] = 'face-'+df['id']+'.png'
image_dir = 'data/faces_processed_64'

predictions = []

print("Auditing images with CLIP...")
# Wrapping the loop in torch.no_grad() speeds up inference and saves VRAM
with torch.no_grad():
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(image_dir, row['id'])
        
        try:
            # CLIP expects PIL images, not OpenCV numpy arrays
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            predictions.append(-1)
            continue
            
        # 2. Process the image and text prompts simultaneously
        inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(device)
        
        # 3. Get the similarity scores
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image 
        probs = logits_per_image.softmax(dim=1) # Convert scores to probabilities
        
        # Determine the winning class (0 for no glasses, 1 for glasses)
        predicted_class = torch.argmax(probs, dim=1).item()
        predictions.append(predicted_class)

# Add the CLIP predictions to your dataframe
df['clip_prediction'] = predictions

df.to_csv('outputs/train_clip.csv')

print("Audit complete.")

#%%
# Find EVERY instance where the original label disagrees with the CLIP prediction
# This automatically captures both False Positives and False Negatives
all_suspects = df[df['glasses'] != df['clip_prediction']].copy()

print(f"Total suspicious labels found: {len(all_suspects)}")

# Look at the breakdown of the errors
false_positives = all_suspects[all_suspects['glasses'] == 1]
false_negatives = all_suspects[all_suspects['glasses'] == 0]

print(f"CSV says 'Glasses', but CLIP says 'No': {len(false_positives)}")
print(f"CSV says 'No Glasses', but CLIP says 'Yes': {len(false_negatives)}")

# Pass this combined list into your manual correction tool
# easy_correct(all_suspects, batch_size=12)