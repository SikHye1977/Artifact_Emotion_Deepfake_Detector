"""Preserve the existing OR rule to isolate the emotion-input change."""
from src.fusion.hierarchical_score_fusion.hierarchical_score_fusion import probabilistic_or as score_or


def fuse(artifact_video, artifact_audio, emotion_video, emotion_audio):
    artifact=score_or(artifact_video,artifact_audio)
    emotion=score_or(emotion_video,emotion_audio)
    return dict(artifact_only=artifact,emotion_only=emotion,hierarchical=score_or(artifact,emotion))
