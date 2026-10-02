import numpy as np
import torch

from pathlib import Path
from torchvision import transforms
from hsemotion.facial_emotions import HSEmotionRecognizer
from src.utils.local_paths import load_local_paths, project_path, file_sha256

HSE_SHA256 = '47c1423f3e6f50e3750bf7b0eda7db947c9ce0c2637e1766bf2187eddc652b17'


def hsemotion_checkpoint():
    return project_path(load_local_paths().get('hsemotion_checkpoint') or
                        'checkpoints/pretrained/hsemotion/enet_b0_8_best_afew.pt')


class LocalHSEmotionRecognizer(HSEmotionRecognizer):
    """HSEmotion 0.3.0 preprocessing/inference with an explicit verified local model.

    The upstream constructor downloads to the user's home. Its initialization is
    reproduced here for the project's fixed enet_b0_8_best_afew model only.
    Inference methods remain inherited from HSEmotion.
    """
    def __init__(self, path, device):
        path = Path(path)
        if file_sha256(path) != HSE_SHA256:
            raise ValueError('HSEmotion checkpoint does not match the recorded official weight')
        self.device = device
        self.is_mtl = False
        self.idx_to_class = dict(enumerate([
            'Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']))
        self.img_size = 224
        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        model = torch.load(path, map_location='cpu', weights_only=False)
        classifier = model.classifier[0] if isinstance(model.classifier, torch.nn.Sequential) else model.classifier
        self.classifier_weights = classifier.weight.detach().cpu().numpy()
        self.classifier_bias = classifier.bias.detach().cpu().numpy()
        model.classifier = torch.nn.Identity()
        self.model = model.to(device).eval()



class EmotionExtractor:
    def __init__(self, device: str = "cpu"):
        self.model_name = "enet_b0_8_best_afew"
        self.checkpoint_path = hsemotion_checkpoint()
        self.recognizer = LocalHSEmotionRecognizer(self.checkpoint_path, device)

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