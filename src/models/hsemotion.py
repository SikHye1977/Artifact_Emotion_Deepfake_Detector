import numpy as np
import torch

from hsemotion.facial_emotions import HSEmotionRecognizer


class EmotionExtractor:
    def __init__(self, device: str = "cpu"):
        self.model_name = "enet_b0_8_best_afew"
        self.recognizer = HSEmotionRecognizer(
            model_name=self.model_name,
            device=device,
        )

        self.class_names = [
            self.recognizer.idx_to_class[i]
            for i in range(len(self.recognizer.idx_to_class))
        ]

    @torch.inference_mode()
    def extract(self, face_rgb: np.ndarray) -> dict:
        if (
            face_rgb.ndim != 3
            or face_rgb.shape[2] != 3
            or face_rgb.dtype != np.uint8
            or face_rgb.size == 0
        ):
            raise ValueError(
                "Expected a nonempty RGB uint8 face image [H, W, 3]"
            )

        emotion, probabilities = self.recognizer.predict_emotions(
            face_rgb,
            logits=False,
        )
        features = self.recognizer.extract_features(face_rgb)

        probabilities = np.asarray(
            probabilities, dtype=np.float32
        ).reshape(-1)

        features = np.asarray(
            features, dtype=np.float32
        ).reshape(-1)

        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite emotion probabilities")

        if not np.isfinite(features).all():
            raise ValueError("Non-finite emotion features")

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("Invalid emotion probabilities")

        return {
            "emotion": emotion,
            "probabilities": probabilities,
            "features": features,
        }