# 학습/검증 중 지표 누적 및 모델 저장/로딩 유틸리티

import torch
from tqdm import tqdm

class AverageMeter:
    """
    손실(loss) 등의 지표를 배치 단위로 누적하고 평균을 계산하는 헬퍼 클래스
    """
    def __init__(self):
        self.sum = 0.0    # 누적 값의 합
        self.count = 0    # 누적된 샘플 수

    def update(self, val, n=1):
        """
        값(val)을 n개만큼 업데이트
        :param val: 배치에서 계산된 평균 loss 값
        :param n: 해당 배치의 샘플 수
        """
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        """ 누적된 값의 평균 반환 """
        return self.sum / self.count if self.count else 0


def save_model(model, path: str):
    """
    모델의 파라미터를 지정된 경로(path)에 저장
    :param model: torch.nn.Module 객체
    :param path: 저장할 파일 경로
    """
    torch.save(model.state_dict(), path)


def load_model(model, path: str, device=None):
    """
    저장된 파라미터를 모델에 로드 후 반환
    :param model: torch.nn.Module 객체
    :param path: 모델 파일 경로
    :param device: 로드 시 사용할 디바이스 (None이면 CPU)
    """
    state = torch.load(path, map_location=device or torch.device('cpu'))
    model.load_state_dict(state)
    return model