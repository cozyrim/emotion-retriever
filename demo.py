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
model.to(CFG.device).eval()
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
    for fn in tqdm(unique_imgs['image'], desc="Embedding Images"): # 이미지 불러와 전처리
        img = Image.open(os.path.join(CFG.image_path, fn)).convert("RGB")
        arr = np.array(img)
        tensor = (
            torch.tensor(get_transforms(train=False)(image=arr)["image"])
            .permute(2,0,1)
            .unsqueeze(0)
            .float()
            .to(CFG.device)
        )
        with torch.no_grad(): # 이미지 feature 뽑기 (추론모드)
            feat = model.image_encoder(tensor)
            embs.append(model.image_proj(feat).cpu().numpy().flatten())
    image_embeddings = np.vstack(embs)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(EMB_FILE, image_embeddings)
    unique_imgs.to_csv(IMG_FILE, index=False)

# 3) 검색 함수 (Top-6)
def search(caption: str):
    enc = tokenizer( # 텍스트 토크나이징
        caption,
        padding="max_length", truncation=True,
        max_length=CFG.max_length,
        return_tensors="pt"
    )
    with torch.no_grad(): # 텍스트 임베딩 만들기
        txt_feat = model.text_encoder(
            enc.input_ids.to(CFG.device),
            enc.attention_mask.to(CFG.device)
        )
        txt_emb = model.text_proj(txt_feat).cpu().numpy().flatten() # 이미지 임베딩들과 유사도 계산
    sims = image_embeddings @ txt_emb / (
        np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(txt_emb)
    )
    idxs = np.argsort(-sims)[:6] # 유사도 기준으로 Top-6 이미지 선택
    return [
        Image.open(os.path.join(CFG.image_path, unique_imgs['image'][i])) # 실제 이미지 파일 열기
        for i in idxs
    ]

# 4) Blocks 레이아웃 + 커스텀 CSS
custom_css = """
/* ─ 전체 중앙 정렬 & 최대 폭 제한 ─ */
.gradio-container {
  max-width: 900px !important;
  margin: 0 auto;
}

/* ─ 제목 크게 ─ */
#title {
  font-size: 10rem !important;
  text-align: center;
  margin-top: 1rem;
  margin-bottom: 0.5rem;
}

/* ─ 캡션+버튼 세로 스택, 가운데 정렬 ─ */
#search-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 2rem;
  margin-bottom: 1.5rem;
}
/* 버튼 살짝 아래로 */
#search-btn { margin-top: 12px; }

/* 입력창 둥글게 + 그림자 */
#txt-input textarea {
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* 버튼 둥글게 + 그림자 */
#search-btn {
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* ─ 갤러리 3열×2행 ─ */
#gallery {
  height: 480px;          /* 2행 분량 높이 고정 */
  overflow-y: hidden;     /* 스크롤 없이 모두 보이도록 */
}
#gallery .wrap {
  padding: 6px;
}
#gallery img {
  border-radius: 8px;
  object-fit: cover;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* ─ Footer 중앙 정렬 ─ */
#footer {
  text-align: center;
  color: #888;
  margin-top: 1.5rem;
}
"""

with gr.Blocks(theme="gstaff/xkcd", css=custom_css) as demo:
    gr.Markdown("🎭 **표정 설명 기반 이미지 검색**", elem_id="title")

    # 캡션 입력 박스 아래에 검색 버튼 세로 스택
    with gr.Column(elem_id="search-col"):
        txt = gr.Textbox(
            label="Caption 입력",
            placeholder="예: happy smiling face",
            lines=2,
            elem_id="txt-input"
        )
        btn = gr.Button(
            "검색",
            variant="primary",
            elem_id="search-btn"
        )

    # 3열×2행 갤러리, elem_id 지정
    gallery = gr.Gallery(
        label="검색 결과",
        columns=3,
        elem_id="gallery"
    )
    btn.click(search, inputs=txt, outputs=gallery)

    gr.Markdown("Powered by CLIP & DistilBERT", elem_id="footer")

if __name__ == "__main__":
    demo.launch()
