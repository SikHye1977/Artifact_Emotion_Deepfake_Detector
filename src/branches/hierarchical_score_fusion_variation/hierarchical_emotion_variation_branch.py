"""Two independently supervised local-variation heads followed by score OR."""
from torch import nn
from src.models.hierarchical_score_fusion_variation.variation_head import VariationHead
from src.fusion.hierarchical_score_fusion_variation.hierarchical_variation_fusion import score_or


class HierarchicalEmotionVariationBranch(nn.Module):
    def __init__(self, audio_dim=27):
        super().__init__()
        self.video_head=VariationHead('video')
        self.audio_head=VariationHead('audio',input_dim=audio_dim)

    def forward(self, video_variation, audio_variation):
        lv,la=self.video_head(video_variation),self.audio_head(audio_variation)
        ev,ea=lv.sigmoid(),la.sigmoid()
        return dict(video_logits=lv,audio_logits=la,video_score=ev,audio_score=ea,
                    score_emotion=score_or(ev,ea))
