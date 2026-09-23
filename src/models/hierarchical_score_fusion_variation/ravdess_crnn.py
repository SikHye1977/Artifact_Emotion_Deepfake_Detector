"""User-provided CNN+GRU architecture, preserving original state_dict keys."""
import torch
from torch import nn

class AudioEmotionCRNN(nn.Module):
    """RAVDESS 학습된 사전학습 모델 (구조 변경 없음)"""
    def __init__(self, num_classes: int = 8):
        super().__init__()
        import torchaudio

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.gru = nn.GRU(input_size=512, hidden_size=128,
                          num_layers=1, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor):
        return self.forward_with_feat(x)[1]

    def forward_with_feat(self, x: torch.Tensor):
        if self.training:
            x = x + torch.randn_like(x) * 1e-6

        x = self.mel_spec(x)
        x = torch.clamp(x, min=1e-5)
        x = self.amplitude_to_db(x)

        x = self.cnn(x)
        B, C, F_dim, T_dim = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T_dim, C * F_dim)

        gru_out, hn = self.gru(x)
        feat = hn.squeeze(0)
        logit = self.classifier(feat)
        return feat, logit



RAVDESS_CLASSES = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

class FrozenRAVDESSRecognizer(nn.Module):
    """Load the ORIGINAL emotion-only checkpoint, never a fake/real fine-tuned head."""
    def __init__(self, checkpoint, *, class_names, device='cpu'):
        super().__init__()
        if len(class_names)!=8 or set(class_names)!=set(RAVDESS_CLASSES):
            raise ValueError('Require the eight RAVDESS classes in checkpoint training order')
        self.class_names=list(class_names)
        self.backbone=AudioEmotionCRNN(num_classes=8)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not state or not all(isinstance(v,torch.Tensor) for v in state.values()):
            raise ValueError('Expected original emotion-only raw state_dict; deepfake checkpoints are not accepted')
        self.backbone.load_state_dict(state,strict=True)
        self.requires_grad_(False)
        self.to(device=device,dtype=torch.float32)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, waveform):
        if waveform.ndim!=3 or waveform.shape[1:]!=(1,48000) or not len(waveform):
            raise ValueError('Expected mono 16kHz waveform [B,1,48000]')
        if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
            raise ValueError('Invalid waveform')
        device=next(self.backbone.parameters()).device
        with torch.autocast(device_type=device.type,enabled=False):
            feat,logits=self.backbone.forward_with_feat(waveform.to(device=device,dtype=torch.float32))
            probabilities=torch.softmax(logits,dim=-1)
        if logits.shape!=(len(waveform),8) or not torch.isfinite(logits).all():
            raise ValueError('Invalid emotion logits')
        return dict(features=feat,logits=logits,probabilities=probabilities)
