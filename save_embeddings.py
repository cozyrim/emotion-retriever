# save_embeddings.py

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DistilBertTokenizer
from config import CFG
from model import CLIPModel
from utils import load_model
from dataset import get_transforms

# 1) 모델 로드
model = CLIPModel()
model = load_model(model, "model_final.pth", device=CFG.device)
model.to(CFG.device)
model.eval()

# 2) 캡션 CSV → 이미지 리스트
df = pd.read_csv(CFG.captions_path)
unique_imgs = df[['image']].drop_duplicates().reset_index(drop=True)

# 3) 임베딩 계산
image_embeddings = []
for fn in tqdm(unique_imgs['image'], desc="Compute Embeddings"):
    img = Image.open(os.path.join(CFG.image_path, fn)).convert("RGB")
    arr = np.array(img)
    tensor = torch.tensor(get_transforms(train=False)(image=arr)["image"]) \
               .permute(2,0,1).unsqueeze(0).float().to(CFG.device)
    with torch.no_grad():
        feat = model.image_encoder(tensor)
        emb = model.image_proj(feat).cpu().numpy().flatten()
    image_embeddings.append(emb)

image_embeddings = np.vstack(image_embeddings)

# 4) 디스크에 저장
os.makedirs("cache", exist_ok=True)
np.save("cache/image_embeddings.npy", image_embeddings)
unique_imgs.to_csv("cache/image_filenames.csv", index=False)

print("Saved embeddings to cache/image_embeddings.npy")
print("Saved filenames to cache/image_filenames.csv")
