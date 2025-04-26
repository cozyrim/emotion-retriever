# config.py: 학습 및 추론에 사용될 설정 정보를 담은 파일

import os
import torch

class CFG:
    # 디버깅 모드 여부
    debug = False
    # 이미지 파일들이 위치한 폴더 경로 (환경 변수로 설정하거나 기본값 사용)
    image_path = os.getenv("IMAGE_PATH", "./images")
    # 캡션 CSV 파일 경로 (환경 변수로 설정하거나 기본값 사용)
    captions_path = os.getenv("CAPTIONS_PATH", "./captions.csv")

    # 학습 관련 하이퍼파라미터
    batch_size = 16                    # 학습 시 배치 크기
    num_workers = 4                    # DataLoader에 사용할 워커 프로세스 수
    head_lr = 1e-3                     # projection head 학습률
    image_encoder_lr = 1e-4            # 이미지 인코더 학습률
    text_encoder_lr = 1e-5             # 텍스트 인코더 학습률
    weight_decay = 1e-3                # 옵티마이저 가중치 감쇠
    epochs = 30                         # 총 학습 epoch 수

    # ——— 여기부터 교체 ———
    # 학습에 사용할 디바이스: MPS → CUDA → CPU 순으로 자동 선택 (Apple Silicon M1/M2/M3/M4 지원)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # ——————————————————

    # 모델 구조 설정
    model_name = "resnet50"                         # 이미지 인코더 backbone
    text_encoder_model = "distilbert-base-uncased"  # 텍스트 인코더 모델
    pretrained = True                               # 사전학습 가중치 사용 여부
    trainable = True                                # 인코더 파라미터를 미세조정할지 여부
    temperature = 1.0                               # contrastive loss 온도 파라미터

    # 입력 크기 및 차원 설정
    size = 224              # 이미지 리사이즈 크기
    max_length = 200        # 텍스트 토크나이징 최대 길이
    projection_dim = 256    # 이미지/텍스트 임베딩 차원
    text_embedding = 768    # 텍스트 인코더 hidden size
    dropout = 0.1           # projection head 드롭아웃 비율

    # ID별로 balance_df에서 추출할 샘플 수
    samples_per_id = 2500