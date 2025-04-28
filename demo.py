import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import gradio as gr
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

# 2) 임베딩 캐시 로드/생성
CACHE_DIR = "cache"
EMB_FILE = os.path.join(CACHE_DIR, "image_embeddings.npy")
IMG_FILE = os.path.join(CACHE_DIR, "image_filenames.csv")
if os.path.exists(EMB_FILE) and os.path.exists(IMG_FILE):
    image_embeddings = np.load(EMB_FILE)
    unique_imgs = pd.read_csv(IMG_FILE)
else:
    df = pd.read_csv(CFG.captions_path)
    unique_imgs = df[['image']].drop_duplicates().reset_index(drop=True)
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
    image_embeddings = np.vstack(embs)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(EMB_FILE, image_embeddings)
    unique_imgs.to_csv(IMG_FILE, index=False)

# 3) 검색 함수 (고정 Top-6)
def search(caption: str):
    enc = tokenizer(caption, padding="max_length", truncation=True,
                     max_length=CFG.max_length, return_tensors="pt")
    input_ids = enc.input_ids.to(CFG.device)
    attention_mask = enc.attention_mask.to(CFG.device)
    with torch.no_grad():
        txt_feat = model.text_encoder(input_ids, attention_mask)
        txt_emb = model.text_proj(txt_feat).cpu().numpy().flatten()
    sims = image_embeddings @ txt_emb / (
        np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(txt_emb)
    )
    idxs = np.argsort(-sims)[:6]
    imgs = [Image.open(os.path.join(CFG.image_path, unique_imgs['image'][i])) for i in idxs]
    return imgs

# 4) Blocks 레이아웃으로 UI 구성 및 스타일링
with gr.Blocks(theme="gstaff/xkcd") as demo:
    gr.Markdown("# 🎭 표정 설명 기반 이미지 검색", elem_id="title")
    with gr.Row():
        txt = gr.Textbox(label="Caption 입력", placeholder="예: happy smiling face", lines=2)
        btn = gr.Button("검색", variant="primary")
    gallery = gr.Gallery(label="검색 결과", columns=6, height="auto")
    btn.click(search, inputs=txt, outputs=gallery)
    gr.HTML("<div style='margin-top:20px; color:gray; font-size:12px;'>Powered by CLIP & DistilBERT</div>")

if __name__ == "__main__":
    demo.launch()
