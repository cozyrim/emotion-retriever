# PyTorch Dataset, DataLoader 및 데이터 불균형 처리 함수 정의

import os
import cv2
import torch
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from config import CFG

class CLIPDataset(Dataset):
    """
    이미지 파일 경로와 캡션 정보를 이용해 학습・검증 데이터셋을 생성
    """
    def __init__(self, df: pd.DataFrame, transforms: A.Compose, tokenizer: DistilBertTokenizer):
        self.df = df.reset_index(drop=True)  # 인덱스 재설정
        self.transforms = transforms         # Albumentations Transform
        self.tokenizer = tokenizer           # Hugging Face 토크나이저

    def __len__(self):
        # 전체 샘플 개수 반환
        return len(self.df)

    def __getitem__(self, idx):
        """
        인덱스(idx)에 해당하는 샘플을 반환
        :return: dict(image, input_ids, attention_mask)
        """
        row = self.df.iloc[idx]
        # 이미지 로드 및 전처리
        img_path = os.path.join(CFG.image_path, row.image)
        image = cv2.imread(img_path)                           # BGR 로드
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         # RGB 변환
        image = self.transforms(image=image)["image"]         # transform 적용
        image = torch.tensor(image).permute(2,0,1).float()     # (H,W,C)->(C,H,W)

        # 캡션 토크나이징
        encoding = self.tokenizer(
            row.caption,
            padding="max_length",
            truncation=True,
            max_length=CFG.max_length,
            return_tensors="pt"
        )
        input_ids = encoding.input_ids.squeeze(0)              # (1, L)->(L)
        attention_mask = encoding.attention_mask.squeeze(0)

        return {"image": image, "input_ids": input_ids, "attention_mask": attention_mask}


def get_transforms(train: bool = True) -> A.Compose:
    """
    학습/검증 모드에 따른 이미지 전처리(transform) 반환
    :param train: True이면 학습용, False면 검증용
    """
    base = [
        A.Resize(CFG.size, CFG.size),      # 크기 조정
        A.Normalize(max_pixel_value=255.0)  # 픽셀 정규화
    ]
    return A.Compose(base)


def build_loaders(df_train: pd.DataFrame, df_valid: pd.DataFrame, tokenizer: DistilBertTokenizer):
    """
    DataLoader 생성
    :return: train_loader, valid_loader
    """
    train_ds = CLIPDataset(df_train, get_transforms(train=True), tokenizer)
    valid_ds = CLIPDataset(df_valid, get_transforms(train=False), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers) # DataLoader는 Pytorch에서 
    valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers) # 데이터를 배치 단위로 묶어주는 역할
    return train_loader, valid_loader


def balance_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 id별로 샘플 수를 동일하게 맞춰 불균형 해소
    :param df: 원본 DataFrame (id, filename, caption 포함)
    """
    dfs = []
    for i in sorted(df.id.unique()):
        df_i = df[df.id == i]
        if len(df_i) >= CFG.samples_per_id:
            # 충분한 샘플이면 비복원 샘플링
            dfs.append(df_i.sample(CFG.samples_per_id, replace=False))
        else:
            # 부족하면 복원 샘플링
            dfs.append(df_i.sample(CFG.samples_per_id, replace=True))
    # 섞어서 반환
    return pd.concat(dfs).sample(frac=1).reset_index(drop=True)