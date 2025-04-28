# train.py: 데이터 로드부터 모델 저장(resume)까지 관리하는 스크립트

import os  # 운영체제 기능 (파일 경로 등) 사용
import glob  # 특정 패턴의 파일 경로 검색
import re  # 정규 표현식 사용
import pandas as pd  # 데이터 처리용 (CSV 로드 등)
import torch  # PyTorch 메인 패키지
from sklearn.model_selection import train_test_split  # 데이터 분할 함수
from torch.optim import AdamW  # AdamW 옵티마이저
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 학습률 스케줄러
from tqdm import tqdm  # 진행률 표시
from transformers import DistilBertTokenizer  # 텍스트 토큰화기

from config import CFG  # 환경설정 상수
from dataset import build_loaders, balance_df  # 데이터 로더 및 불균형 해소 함수
from model import CLIPModel  # 모델 클래스
from utils import AverageMeter, save_model  # 손실 누적기 및 모델 저장 유틸


def find_latest_checkpoint():
    """
    checkpoint_epoch_*.pth 파일 중 가장 최신 에폭의 파일 경로를 반환.
    없으면 None 반환
    """
    # 현재 디렉토리에서 checkpoint_epoch_*.pth 패턴에 맞는 파일 목록 조회
    files = glob.glob("checkpoint_epoch_*.pth")
    if not files:
        return None
    epochs = []  # (epoch 번호, 파일 경로) 쌍을 저장할 리스트
    for f in files:
        # 파일명에서 epoch 숫자 추출
        m = re.search(r"checkpoint_epoch_(\d+)\.pth", f)
        if m:
            epochs.append((int(m.group(1)), f))
    if not epochs:
        return None
    # 에폭 번호가 가장 큰 파일 경로 반환
    latest = max(epochs, key=lambda x: x[0])[1]
    return latest


def main():
    # 1) 이전에 저장된 체크포인트 탐지
    resume_path = find_latest_checkpoint()
    start_epoch = 1  # 기본 시작 에폭

    # 2) 데이터 로드 및 불균형 처리
    df = pd.read_csv(CFG.captions_path)  # captions.csv 불러오기
    df_bal = balance_df(df)  # id별 샘플 수 맞춰서 불균형 해소
    # 학습/검증 세트로 분할 (stratify=id, random_state 고정)
    df_train, df_valid = train_test_split(
        df_bal, test_size=0.2, stratify=df_bal.id, random_state=42
    )
    # 텍스트 토크나이저 초기화
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_encoder_model)
    # DataLoader 생성 (train/valid)
    train_loader, valid_loader = build_loaders(df_train, df_valid, tokenizer)

    # 3) 모델·옵티마이저·스케줄러 설정
    model = CLIPModel().to(CFG.device)  # 모델 인스턴스 생성 및 디바이스 할당
    optimizer = AdamW([
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},  # 이미지 인코더 lr
        {"params": model.text_encoder.parameters(),  "lr": CFG.text_encoder_lr},   # 텍스트 인코더 lr
        {"params": list(model.image_proj.parameters()) + list(model.text_proj.parameters()), "lr": CFG.head_lr},  # projection head lr
    ], weight_decay=CFG.weight_decay)
    # 검증 loss 기준으로 lr 조정 스케줄러
    scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=1)

    # 4) 체크포인트 복원 (resume) 로직
    if resume_path:
        ckpt = torch.load(resume_path, map_location=CFG.device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            # 저장된 dict 형태로 복원
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["opt_state"])
            scheduler.load_state_dict(ckpt["sched_state"])
            start_epoch = ckpt["epoch"] + 1  # 다음 에폭부터 학습
            print(f"Resuming from wrapped checkpoint at epoch {start_epoch}")
        else:
            # 과거 방식 (state_dict) 복원
            model.load_state_dict(ckpt)
            # 파일명에서 에폭 번호 추출
            m = re.search(r"checkpoint_epoch_(\d+)\.pth", resume_path)
            if m:
                start_epoch = int(m.group(1)) + 1
            print(f"Resuming from raw state_dict, starting at epoch {start_epoch}")

    # 5) 학습 루프
    for epoch in range(start_epoch, CFG.epochs + 1):
        # 학습 모드
        model.train()
        train_meter = AverageMeter()  # 학습 손실 누적기
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            # 배치 내 모든 tensor를 GPU/CPU로 이동
            batch = {k: v.to(CFG.device) for k, v in batch.items()}

            optimizer.zero_grad()  # 기울기 초기화
            loss = model(batch)   # 순전파 & 손실 계산
            loss.backward()       # 역전파
            optimizer.step()      # 파라미터 업데이트
            # 손실 누적
            train_meter.update(loss.item(), batch["image"].size(0))

        # 검증 모드
        model.eval()
        valid_meter = AverageMeter()  # 검증 손실 누적기
        with torch.no_grad():  # 기울기 계산 비활성화
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch} Valid"):
                batch = {k: v.to(CFG.device) for k, v in batch.items()}
                loss = model(batch)
                valid_meter.update(loss.item(), batch["image"].size(0))

        # 스케줄러 스텝 (검증 loss 기준)
        scheduler.step(valid_meter.avg)
        print(f"Epoch {epoch}: Train Loss={train_meter.avg:.4f}, Valid Loss={valid_meter.avg:.4f}")

        # 6) 체크포인트 저장 (덮어쓰기)
        ckpt_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict()
        }
        save_path = resume_path or f"checkpoint_epoch_{epoch}.pth"
        torch.save(ckpt_dict, save_path)
        print("Saved checkpoint:", save_path)

    # 7) 최종 모델 저장
    save_model(model, "model_final.pth")
    print("Final model saved: model_final.pth")


if __name__ == "__main__":
    main()
