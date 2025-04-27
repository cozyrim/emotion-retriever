# train.py: 데이터 로드부터 모델 저장(resume)까지

import os
import glob
import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import DistilBertTokenizer

from config import CFG
from dataset import build_loaders, balance_df
from model import CLIPModel
from utils import AverageMeter, save_model

def find_latest_checkpoint():
    """
    checkpoint_epoch_*.pth 파일 중 가장 최신 에폭의 파일 경로를 반환.
    없으면 None을 반환합니다.
    """
    files = glob.glob("checkpoint_epoch_*.pth")
    if not files:
        return None
    epochs = []
    for f in files:
        m = re.search(r"checkpoint_epoch_(\d+)\.pth", f)
        if m:
            epochs.append((int(m.group(1)), f))
    if not epochs:
        return None
    # 가장 큰 에폭 번호의 파일 경로 반환
    return max(epochs, key=lambda x: x[0])[1]

def main():
    # 1) 최신 체크포인트 자동 감지
    resume_path = find_latest_checkpoint()
    start_epoch = 1

    # 2) 데이터 로드 및 전처리
    df = pd.read_csv(CFG.captions_path)
    df_bal = balance_df(df)
    df_train, df_valid = train_test_split(
        df_bal, test_size=0.2, stratify=df_bal.id, random_state=42
    )
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_encoder_model)
    train_loader, valid_loader = build_loaders(df_train, df_valid, tokenizer)

    # 3) 모델·옵티마이저·스케줄러 초기화
    model = CLIPModel().to(CFG.device)
    optimizer = AdamW([
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(),  "lr": CFG.text_encoder_lr},
        {"params": list(model.image_proj.parameters()) + list(model.text_proj.parameters()), "lr": CFG.head_lr},
    ], weight_decay=CFG.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=1)

    # 4) 체크포인트 로드(resume)
    resume_path = find_latest_checkpoint()  # 예: "checkpoint_epoch_11.pth"
    start_epoch = 1

    if resume_path:
        ckpt = torch.load(resume_path, map_location=CFG.device)
        if "model_state" in ckpt:
            # 기존에 추가했던 wrapped checkpoint (model_state, opt_state 등)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["opt_state"])
            scheduler.load_state_dict(ckpt["sched_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resuming from wrapped checkpoint at epoch {start_epoch}")
        else:
            # 구버전: 순수한 state_dict만 있을 때
            model.load_state_dict(ckpt)
            # 파일명에서 epoch 파싱 (예시: checkpoint_epoch_11.pth → 11)
            m = re.search(r"checkpoint_epoch_(\d+)\.pth", resume_path)
            if m:
                start_epoch = int(m.group(1)) + 1
            print(f"Resuming from raw state_dict, starting at epoch {start_epoch}")


    # 5) 학습 루프(start_epoch부터)
    for epoch in range(start_epoch, CFG.epochs + 1):
        model.train()
        train_meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            # (★) 모든 입력을 디바이스로 이동
            batch = {k: v.to(CFG.device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_meter.update(loss.item(), batch["image"].size(0))

        model.eval()
        valid_meter = AverageMeter()
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch} Valid"):
                batch = {k: v.to(CFG.device) for k, v in batch.items()}
                loss = model(batch)
                valid_meter.update(loss.item(), batch["image"].size(0))

        scheduler.step(valid_meter.avg)
        print(f"Epoch {epoch}: Train Loss={train_meter.avg:.4f}, Valid Loss={valid_meter.avg:.4f}")

        # 6) 체크포인트 저장 (덮어쓰기)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict()
        }, resume_path or f"checkpoint_epoch_{epoch}.pth")
        print("Saved checkpoint:", resume_path or f"checkpoint_epoch_{epoch}.pth")

    # 7) 최종 모델 저장
    save_model(model, "model_final.pth")

if __name__ == "__main__":
    main()
