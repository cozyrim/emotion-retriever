# 이미지・텍스트 인코더와 프로젝션 헤드, contrastive loss 정의

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig
from config import CFG

class ImageEncoder(nn.Module):
    """
    ResNet50 backbone을 이용한 이미지 인코더
    """
    def __init__(self):
        super().__init__()
        # num_classes=0, global_pool="avg" -> feature vector 출력
        self.backbone = timm.create_model(
            CFG.model_name, pretrained=CFG.pretrained, num_classes=0, global_pool="avg"
        )
        # 파라미터 미세조정 여부
        for p in self.backbone.parameters():
            p.requires_grad = CFG.trainable

    def forward(self, x):
        # x: (B, C, H, W)
        return self.backbone(x)  # (B, feature_dim)

class TextEncoder(nn.Module):
    """
    DistilBERT 기반 텍스트 인코더 (CLS 토큰 사용)
    """
    def __init__(self):
        super().__init__()
        if CFG.pretrained:
            # 사전학습된 모델 로드
            self.model = DistilBertModel.from_pretrained(CFG.text_encoder_model)
        else:
            self.model = DistilBertModel(DistilBertConfig())
        # 파라미터 미세조정 여부
        for p in self.model.parameters():
            p.requires_grad = CFG.trainable

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # CLS 토큰 벡터 (batch, hidden_dim)
        return out.last_hidden_state[:, 0]

class ProjectionHead(nn.Module):
    """
    임베딩 차원 사영: Linear -> GELU -> Dropout
    """
    def __init__(self, dim_in: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, CFG.projection_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(CFG.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return self.drop(x)

class CLIPModel(nn.Module):
    """
    CLIP 스타일 대조 학습 모델
    """
    def __init__(self):
        super().__init__()
        # 이미지 인코더 및 프로젝션 헤드
        self.image_encoder = ImageEncoder()
        img_dim = self.image_encoder.backbone.num_features  # backbone output dim
        self.image_proj = ProjectionHead(img_dim)
        # 텍스트 인코더 및 프로젝션 헤드
        self.text_encoder = TextEncoder()
        # hidden_size가 필요하지만 없으면 projection_dim fallback
        txt_dim = getattr(CFG, 'text_embedding', CFG.projection_dim)
        self.text_proj = ProjectionHead(txt_dim)
        # contrastive loss 온도
        self.temp = CFG.temperature

    def forward(self, batch):
        # 이미지 임베딩
        img_feat = self.image_encoder(batch["image"])
        img_emb = self.image_proj(img_feat)  # (B, proj_dim)
        # 텍스트 임베딩
        txt_feat = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        txt_emb = self.text_proj(txt_feat)  # (B, proj_dim)
        # 유사도 행렬 계산
        logits = img_emb @ txt_emb.t() / self.temp
        labels = torch.arange(logits.size(0), device=logits.device)
        # 이미지->텍스트, 텍스트->이미지 loss
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_i + loss_t) / 2