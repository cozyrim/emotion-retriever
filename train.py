# 학습 스크립트: 데이터 로드부터 모델 저장까지

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from config import CFG
from dataset import build_loaders, balance_df
from model import CLIPModel
from utils import AverageMeter, save_model
from transformers import DistilBertTokenizer


def main():
    # 1) CSV 파일 로드
    df = pd.read_csv(CFG.captions_path)
    # 2) ID별 balance 처리
    df_bal = balance_df(df)
    # 3) 학습/검증 분할 (id 기준 stratify)
    df_train, df_valid = train_test_split(
        df_bal, test_size=0.2, stratify=df_bal.id, random_state=42
    )
    # 4) 토크나이저
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_encoder_model)
    # 5) DataLoader
    train_loader, valid_loader = build_loaders(df_train, df_valid, tokenizer)
    # 6) 모델 초기화 및 디바이스 할당
    model = CLIPModel().to(CFG.device)
    # 7) 옵티마이저 설정
    optimizer = AdamW([
        {"params": model.image_encoder.parameters(),  "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(),   "lr": CFG.text_encoder_lr},
        {"params": list(model.image_proj.parameters()) + list(model.text_proj.parameters()), "lr": CFG.head_lr},
    ], weight_decay=CFG.weight_decay)
    # 8) 스케줄러 설정
    scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=1)
    # 9) 에폭 루프
    for epoch in range(1, CFG.epochs + 1):
        # 학습
        model.train()
        train_meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_meter.update(loss.item(), batch["image"].size(0))
        # 검증
        model.eval()
        valid_meter = AverageMeter()
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch} Valid"):
                loss = model(batch)
                valid_meter.update(loss.item(), batch["image"].size(0))
        # 스케줄러 스텝
        scheduler.step(valid_meter.avg)
        # 로그 출력
        print(f"Epoch {epoch}: Train Loss={train_meter.avg:.4f}, Valid Loss={valid_meter.avg:.4f}")
        # 체크포인트 저장
        save_model(model, f"checkpoint_epoch_{epoch}.pth")
    # 최종 모델 저장
    save_model(model, "model_final.pth")

if __name__ == "__main__":
    main()