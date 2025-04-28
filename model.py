# 이미지・텍스트 인코더와 프로젝션 헤드, contrastive loss 정의

import torch  # PyTorch 기본 라이브러리
import torch.nn as nn  # 신경망 모듈을 위한 라이브러리
import torch.nn.functional as F  # 함수형 API (활성화 함수, 손실 함수 등)
import timm  # 다양한 이미지 모델을 쉽게 불러오기 위한 라이브러리
from transformers import DistilBertModel, DistilBertConfig  # 텍스트 인코더 모델 및 구성
from config import CFG  # 사용자 정의 설정값 불러오기

# 이미지 인코더 클래스: ResNet50 기반 이미지 특징 추출기
class ImageEncoder(nn.Module):
    """
    ResNet50 backbone을 이용한 이미지 인코더
    """
    def __init__(self):
        super().__init__()
        # num_classes=0, global_pool="avg" -> feature vector 출력
        # timm 라이브러리로 사전학습된 ResNet50 모델 생성
        self.backbone = timm.create_model(
            CFG.model_name, pretrained=CFG.pretrained, num_classes=0, global_pool="avg"
        )
        # 파라미터 미세조정 여부 설정 (True면 학습 가능)
        for p in self.backbone.parameters():
            p.requires_grad = CFG.trainable

    def forward(self, x):
        # x: (B, C, H, W) 형태의 이미지 배치 입력
        return self.backbone(x)  # (B, feature_dim) 형태의 특징 벡터 반환

# 텍스트 인코더 클래스: DistilBERT 기반 텍스트 특징 추출기 (CLS 토큰 사용)
class TextEncoder(nn.Module):
    """
    DistilBERT 기반 텍스트 인코더 (CLS 토큰 사용)
    """
    def __init__(self):
        super().__init__()
        if CFG.pretrained:
            # 사전학습된 DistilBERT 모델 로드
            self.model = DistilBertModel.from_pretrained(CFG.text_encoder_model)
        else:
            # 새로운 DistilBERT 모델 초기화
            self.model = DistilBertModel(DistilBertConfig())
        # 파라미터 미세조정 여부 설정
        for p in self.model.parameters():
            p.requires_grad = CFG.trainable

    def forward(self, input_ids, attention_mask):
        # 모델에 input_ids와 attention_mask 입력하여 출력 획득
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # CLS 토큰 벡터 (batch, hidden_dim) 추출하여 반환
        return out.last_hidden_state[:, 0]

# 임베딩 차원 사영을 위한 프로젝션 헤드: Linear -> GELU -> Dropout
class ProjectionHead(nn.Module):
    """
    임베딩 차원 사영: Linear -> GELU -> Dropout
    """
    def __init__(self, dim_in: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, CFG.projection_dim)  # 입력 차원 -> 출력 차원 선형 변환
        self.act = nn.GELU()  # 활성화 함수
        self.drop = nn.Dropout(CFG.dropout)  # 드롭아웃 적용

    def forward(self, x):
        x = self.fc(x)  # 선형 변환
        x = self.act(x)  # 활성화 함수 적용
        return self.drop(x)  # 드롭아웃 적용 후 반환

# CLIP 스타일 대조 학습 모델 정의
class CLIPModel(nn.Module):
    """
    CLIP 스타일 대조 학습 모델
    """
    def __init__(self):
        super().__init__()
        # 이미지 인코더 및 프로젝션 헤드 생성
        self.image_encoder = ImageEncoder()
        img_dim = self.image_encoder.backbone.num_features  # backbone 출력 차원 획득
        self.image_proj = ProjectionHead(img_dim)  # 이미지 임베딩 변환기 생성
        # 텍스트 인코더 및 프로젝션 헤드 생성
        self.text_encoder = TextEncoder()
        # hidden_size가 필요하지만 없으면 projection_dim fallback
        txt_dim = getattr(CFG, 'text_embedding', CFG.projection_dim)
        self.text_proj = ProjectionHead(txt_dim)  # 텍스트 임베딩 변환기 생성
        # contrastive loss 온도 파라미터 설정
        self.temp = CFG.temperature

    def forward(self, batch):
        # 이미지 임베딩 생성
        img_feat = self.image_encoder(batch["image"])  # 이미지 특징 추출
        img_emb = self.image_proj(img_feat)  # 프로젝션 헤드 통과 (B, proj_dim)
        # 텍스트 임베딩 생성
        txt_feat = self.text_encoder(batch["input_ids"], batch["attention_mask"])  # 텍스트 특징 추출
        txt_emb = self.text_proj(txt_feat)  # 프로젝션 헤드 통과 (B, proj_dim)
        # 이미지-텍스트 임베딩 간 유사도 행렬 계산 (내적 후 온도 조절)
        logits = img_emb @ txt_emb.t() / self.temp
        labels = torch.arange(logits.size(0), device=logits.device)  # 정답 레이블 생성
        # 이미지->텍스트 방향의 크로스 엔트로피 손실 계산
        loss_i = F.cross_entropy(logits, labels)
        # 텍스트->이미지 방향의 크로스 엔트로피 손실 계산
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_i + loss_t) / 2  # 양방향 손실 평균 반환