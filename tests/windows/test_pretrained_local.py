import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from src.models.hierarchical_score_fusion_variation import emotion2vec_seed as seed


def test_installed_snapshot_is_offline_and_rejects_tampering(tmp_path, monkeypatch):
    commit = 'a' * 40
    root = tmp_path / commit
    root.mkdir()
    for name in ('model.pt', 'config.yaml'):
        (root / name).write_bytes(name.encode())
    metadata = {'revision': commit, 'files': [
        {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in root.iterdir()]}
    provenance = tmp_path / 'checkpoints/pretrained/download_provenance.json'
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps(metadata), encoding='utf-8')
    monkeypatch.setattr(seed, 'load_local_paths', lambda: {'emotion2vec_snapshot': str(root)})
    monkeypatch.setattr(seed, 'project_path', lambda x: Path(x) if Path(x).is_absolute() else tmp_path / x)
    with patch('huggingface_hub.HfApi', side_effect=AssertionError('Network forbidden')):
        assert seed.download_seed() == (str(root), commit)
        with pytest.raises(ValueError, match='revision differs'):
            seed.download_seed(revision='b' * 40)
        (root / 'model.pt').write_bytes(b'tampered')
        with pytest.raises(ValueError, match='hash mismatch'):
            seed.download_seed()


def test_hsemotion_rejects_wrong_weight_before_unpickling(tmp_path):
    from src.models.hsemotion import LocalHSEmotionRecognizer
    wrong = tmp_path / 'wrong.pt'
    wrong.write_bytes(b'not a trusted checkpoint')
    with patch('torch.load', side_effect=AssertionError('Must not unpickle')):
        with pytest.raises(ValueError, match='does not match'):
            LocalHSEmotionRecognizer(wrong, 'cpu')
