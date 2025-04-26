import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import DistilBertTokenizer
from config import CFG
from dataset import build_loaders, balance_df
from model import CLIPModel
from utils import load_model, AverageMeter

# 학습된 모델 성능 평가: Loss, Recall@1, Recall@5 출력

def evaluate():
    # 1) CSV 로드 및 불균형 처리
    df = pd.read_csv(CFG.captions_path)
    df_bal = balance_df(df)
    # 2) 검증 세트 분할 (검증만 사용)
    _, df_valid = train_test_split(
        df_bal, test_size=0.2, stratify=df_bal.id, random_state=42
    )
    # 3) 토크나이저 및 DataLoader 생성
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_encoder_model)
    _, valid_loader = build_loaders(df_bal, df_bal, tokenizer)

    # 4) 모델 로드 및 준비
    model = CLIPModel().to(CFG.device)
    model = load_model(model, "model_final.pth", device=CFG.device)
    model.eval()

    # 5) 평가 지표 초기화
    loss_meter = AverageMeter()
    correct1, correct5, total = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Evaluating"):
            # 디바이스 이동
            imgs = batch["image"].to(CFG.device)
            input_ids = batch["input_ids"].to(CFG.device)
            attention_mask = batch["attention_mask"].to(CFG.device)

            # 손실 계산
            batch_dict = {"image": imgs, "input_ids": input_ids, "attention_mask": attention_mask}
            loss = model(batch_dict)
            loss_meter.update(loss.item(), imgs.size(0))

            # 임베딩 계산 및 유사도
            img_emb = model.image_proj(model.image_encoder(imgs))
            txt_emb = model.text_proj(model.text_encoder(input_ids, attention_mask))
            sims = img_emb @ txt_emb.t()
            batch_size = sims.size(0)
            labels = torch.arange(batch_size, device=CFG.device)

            # Recall@1
            preds1 = sims.argmax(dim=1)
            correct1 += (preds1 == labels).sum().item()

            # Recall@5
            top5 = sims.topk(5, dim=1).indices
            match5 = top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            correct5 += match5

            total += batch_size

    # 최종 결과 출력
    print(f"Average Loss: {loss_meter.avg:.4f}")
    print(f"Recall@1: {correct1/total:.4f}")
    print(f"Recall@5: {correct5/total:.4f}")

if __name__ == "__main__":
    evaluate()
