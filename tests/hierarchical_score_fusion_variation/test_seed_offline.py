import json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import download_seed,sha256,REPO
class OfflineTests(unittest.TestCase):
    def test_offline_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.pt').write_bytes(b'weight');(root/'config.yaml').write_text('model: seed')
            meta={'model':{'repo':REPO,'revision':'a'*40,'files_sha256':{n:sha256(root/n) for n in ['model.pt','config.yaml']}}}
            manifest=root/'manifest.json';manifest.write_text(json.dumps(meta))
            hub=types.ModuleType('huggingface_hub');hub.HfApi=Mock(side_effect=AssertionError('Network API forbidden'));hub.snapshot_download=Mock(return_value=tmp)
            with patch.dict('sys.modules',{'huggingface_hub':hub}):
                self.assertEqual(download_seed(evaluation_manifest=manifest),(tmp,'a'*40))
                hub.snapshot_download.assert_called_with(REPO,revision='a'*40,local_files_only=True,token=False)
                (root/'model.pt').write_bytes(b'changed')
                with self.assertRaises(ValueError):download_seed(evaluation_manifest=manifest)
    def test_online_anonymous(self):
        hub=types.ModuleType('huggingface_hub');api=Mock();api.model_info.return_value.sha='b'*40
        hub.HfApi=Mock(return_value=api);hub.snapshot_download=Mock(return_value='/fake/snapshot')
        with patch.dict('sys.modules',{'huggingface_hub':hub}):download_seed()
        api.model_info.assert_called_once_with(REPO,revision='main',token=False)
        hub.snapshot_download.assert_called_once_with(REPO,revision='b'*40,token=False)
