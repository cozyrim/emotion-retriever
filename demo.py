# Gradio를 이용한 검색 GUI 구현

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import gradio as gr
from gradio.themes import XKCD
from tqdm import tqdm
from transformers import DistilBertTokenizer
from config import CFG
from model import CLIPModel
from utils import load_model
from dataset import get_transforms

# 1) 모델 및 토크나이저 로드
model = CLIPModel()
model = load_model(model, "model_final.pth", device=CFG.device)
model.to(CFG.device)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_encoder_model)

# 2) 이미지 파일 목록 및 임베딩 계산
df = pd.read_csv(CFG.captions_path)
unique_imgs = df[['image']].drop_duplicates().reset_index(drop=True)

def build_image_bank():
    """
    모든 이미지에 대해 임베딩 계산
    """
    embs = []
    for fn in tqdm(unique_imgs['image'], desc="Embedding Images"):
        img = Image.open(os.path.join(CFG.image_path, fn)).convert("RGB")
        arr = np.array(img)
        tensor = torch.tensor(get_transforms(train=False)(image=arr)["image"]) \
                 .permute(2,0,1).unsqueeze(0).float().to(CFG.device)
        with torch.no_grad():
            feat = model.image_encoder(tensor)
            emb = model.image_proj(feat).cpu().numpy().flatten()
        embs.append(emb)
    return np.vstack(embs)

image_embeddings = build_image_bank()

def search(caption: str, top_k: int = 5):
    """
    입력 텍스트에 유사도가 높은 이미지 top_k개 반환
    """
    enc = tokenizer(caption, padding="max_length", truncation=True, max_length=CFG.max_length, return_tensors="pt")
    input_ids = enc.input_ids.to(CFG.device)
    attention_mask = enc.attention_mask.to(CFG.device)
    with torch.no_grad():
        txt_feat = model.text_encoder(input_ids, attention_mask)
        txt_emb = model.text_proj(txt_feat).cpu().numpy().flatten()
    sims = image_embeddings @ txt_emb / (np.linalg.norm(image_embeddings,axis=1) * np.linalg.norm(txt_emb))
    idxs = np.argsort(-sims)[:top_k]
    imgs = [Image.open(os.path.join(CFG.image_path, unique_imgs.filename[i])) for i in idxs]
    return imgs

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=search,
    inputs=[gr.Textbox(label="Caption 입력"), gr.Slider(1,10,value=5,label="Top K")],
    outputs=gr.Gallery(label="검색 결과"),
    title="표정 설명 기반 이미지 검색",
    theme=XKCD()
)

if __name__ == "__main__":
    iface.launch()
