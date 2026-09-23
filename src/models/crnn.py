"""AMSDF ACRNN을 이용한 고정 오디오 감정 특징 추출."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_FILE = (
    PROJECT_ROOT / "third_party/amsdf_acrnn/ACRNN.py"
)


def load_acrnn_class(source_file: Path):
    """다른 모델 모듈과 이름이 충돌하지 않도록 파일 경로로 불러옵니다."""
    source_file = Path(source_file).resolve()

    if not source_file.is_file():
        raise FileNotFoundError(
            f"ACRNN 소스 파일이 없습니다: {source_file}"
        )

    spec = spec_from_file_location(
        "_vendor_amsdf_acrnn",
        source_file,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"ACRNN 소스를 불러올 수 없습니다: {source_file}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, "acrnn", None)

    if (
        not isinstance(model_class, type)
        or not issubclass(model_class, nn.Module)
    ):
        raise TypeError(
            "소스 파일에 nn.Module을 상속한 acrnn 클래스가 없습니다."
        )

    return model_class


class AudioEmotionExtractor(nn.Module):
    """
    사전학습 ACRNN에서 오디오 구간별 특징을 추출합니다.

    입력:
        acoustic_features: [B, 3, 300, 40]

    출력:
        features: [B, 256]
        weighted_sequence: [B, 150, 256]
        attention: [B, 150]

    감정 분류 head를 사용하지 않으므로 감정 확률은 반환하지 않습니다.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        source_file: str | Path = DEFAULT_SOURCE_FILE,
        device: str = "cpu",
        state_dict_key: str | None = None,
    ):
        super().__init__()

        self.source_file = Path(source_file).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"사전학습 가중치가 없습니다: {self.checkpoint_path}"
            )

        model_class = load_acrnn_class(self.source_file)
        self.backbone = model_class()

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if state_dict_key is None:
            state_dict = checkpoint
        else:
            if (
                not isinstance(checkpoint, Mapping)
                or state_dict_key not in checkpoint
            ):
                raise KeyError(
                    f"checkpoint에 {state_dict_key!r} 키가 없습니다."
                )

            state_dict = checkpoint[state_dict_key]

        if (
            not isinstance(state_dict, Mapping)
            or not state_dict
            or not all(
                isinstance(key, str) and isinstance(value, torch.Tensor)
                for key, value in state_dict.items()
            )
        ):
            raise TypeError(
                "가중치는 state_dict 형태여야 합니다. "
                "중첩된 checkpoint라면 state_dict_key를 지정하세요."
            )

        # 구조가 다른 가중치를 일부만 읽고 진행하지 않습니다.
        self.backbone.load_state_dict(state_dict, strict=True)

        self.backbone.requires_grad_(False)
        self.to(device=device, dtype=torch.float32)
        self.eval()

    def train(self, mode: bool = True):
        """특징 추출기는 항상 평가 모드로 유지합니다."""
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(
        self,
        acoustic_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """전처리된 오디오 배치를 받아 GPU/CPU 텐서를 반환합니다."""
        if not isinstance(acoustic_features, torch.Tensor):
            raise TypeError("입력은 torch.Tensor여야 합니다.")

        if (
            acoustic_features.ndim != 4
            or tuple(acoustic_features.shape[1:]) != (3, 300, 40)
            or acoustic_features.shape[0] == 0
        ):
            raise ValueError(
                "입력 형태는 [B, 3, 300, 40]이어야 합니다."
            )

        if not acoustic_features.is_floating_point():
            raise TypeError("입력은 실수형 텐서여야 합니다.")

        if not torch.isfinite(acoustic_features).all():
            raise ValueError("입력에 NaN 또는 Inf가 있습니다.")

        self.eval()

        model = self.backbone
        device = next(model.parameters()).device

        # 외부에서 AMP를 사용하더라도 여기서는 FP32로 추출합니다.
        with torch.autocast(device_type=device.type, enabled=False):
            x = acoustic_features.to(
                device=device,
                dtype=torch.float32,
            )
            batch_size = x.shape[0]

            # CNN: 원본 ACRNN의 처리 순서를 유지합니다.
            for index in range(1, 7):
                convolution = getattr(model, f"conv{index}")
                x = model.relu(convolution(x))

                if index == 1:
                    x = F.max_pool2d(
                        x,
                        kernel_size=(2, 4),
                        stride=(2, 4),
                    )

                x = model.dropout(x)

            # [B, 256, 150, 10]
            expected_shape = (
                model.L2,
                model.time_step,
                model.p,
            )

            if tuple(x.shape[1:]) != expected_shape:
                raise RuntimeError(
                    f"CNN 출력 형태가 예상과 다릅니다: {tuple(x.shape)}"
                )

            # 원본 코드와 같은 순서로 주파수·채널 차원을 펼칩니다.
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.reshape(-1, model.L2 * model.p)

            x = model.relu(model.bn(model.linear1(x)))
            x = x.reshape(
                batch_size,
                model.time_step,
                model.num_linear,
            )

            # 양방향 LSTM: [B, 150, 256]
            sequence, _ = model.rnn(x)

            # squeeze(-1)로 배치 크기가 1이어도 배치 차원을 보존합니다.
            attention_scores = model.a_fc2(
                model.sigmoid(model.a_fc1(sequence))
            ).squeeze(-1)

            attention = torch.softmax(
                attention_scores,
                dim=1,
            )

            # AMSDF ACRNN의 출력에 해당합니다.
            weighted_sequence = (
                sequence * attention.unsqueeze(-1)
            )

            # 이번 trajectory 분석에서 사용할 구간별 요약입니다.
            features = weighted_sequence.sum(dim=1)

        outputs = {
            "features": features,
            "weighted_sequence": weighted_sequence,
            "attention": attention,
        }

        for name, tensor in outputs.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{name} 출력에 NaN 또는 Inf가 있습니다."
                )

        return outputs

    def extract(self, acoustic_features: torch.Tensor) -> dict:
        """HSEmotion wrapper처럼 NumPy 배열 형태로 반환합니다."""
        outputs = self(acoustic_features)

        return {
            **{
                name: tensor.detach().cpu().numpy()
                for name, tensor in outputs.items()
            },
            "emotion": None,
            "probabilities": None,
        }