import json
import os
from pathlib import Path
import tempfile
import unittest
from src.utils.local_paths import MediaFileVerifier, file_sha256, resolve_media_path


class WindowsPathsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.local=self.root/'media';self.local.mkdir()
        self.config={'media_roots':[{'source':'/linux/FAV','target':str(self.local)}]}
        self.elig=self.root/'eligibility.jsonl';self.elig.write_text('original audit')

    def test_root_boundary_and_traversal(self):
        self.assertEqual(resolve_media_path('/linux/FAV/a.mp4',self.config),self.local/'a.mp4')
        with self.assertRaises(ValueError):resolve_media_path('/linux/FAV/../outside.mp4',self.config)
        if os.name=='nt':
            with self.assertRaises(ValueError):resolve_media_path('/linux/FAV_backup/a.mp4',self.config)
        else:
            self.assertEqual(resolve_media_path('/linux/FAV_backup/a.mp4',self.config),Path('/linux/FAV_backup/a.mp4'))

    def row(self):
        p=self.local/'a.mp4';p.write_bytes(b'original media')
        return dict(dataset='FAV',sample_id='fixed-id',path=str(p),source_path='/linux/FAV/a.mp4',size=p.stat().st_size,mtime_ns=123)

    def make_audit(self,row):
        path=Path(row['path']);st=path.stat();h=file_sha256(path)
        e=dict(dataset='FAV',sample_id=row['sample_id'],source_path=row['source_path'],local_path=str(path),
               source_size=row['size'],source_mtime_ns=row['mtime_ns'],size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=h,source_sha256=h)
        audit=self.root/'relocation.jsonl'
        audit.write_text(json.dumps(dict(format='media_relocation_v1',eligibility_sha256=file_sha256(self.elig)))+'\n'+json.dumps(e)+'\n')
        return dict(self.config,relocation_audit=str(audit))

    def test_relocation_requires_audit(self):
        with self.assertRaisesRegex(ValueError,'requires a verified'):
            MediaFileVerifier(self.elig,self.config).verify(self.row())

    def test_valid_relocation_preserves_source_identity(self):
        row=self.row();config=self.make_audit(row)
        MediaFileVerifier(self.elig,config).verify(row)
        self.assertEqual(row['source_path'],'/linux/FAV/a.mp4')
        self.assertEqual(row['mtime_ns'],123)

    def test_changed_local_media_rejected(self):
        row=self.row();config=self.make_audit(row)
        Path(row['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):MediaFileVerifier(self.elig,config).verify(row)

    def test_content_hash_checked_with_matching_size_mtime(self):
        row=self.row();config=self.make_audit(row);p=Path(row['path']);st=p.stat()
        p.write_bytes(b'x'*st.st_size);os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'hash mismatch'):MediaFileVerifier(self.elig,config).verify(row)

    def test_wrong_eligibility_rejected(self):
        row=self.row();config=self.make_audit(row);self.elig.write_text('changed audit')
        with self.assertRaises(ValueError):MediaFileVerifier(self.elig,config)

    def test_native_mtime_still_checked(self):
        row=self.row();row['source_path']=row['path']
        row['mtime_ns']=Path(row['path']).stat().st_mtime_ns
        verifier=MediaFileVerifier(self.elig,{});verifier.verify(row)
        row['mtime_ns']-=1
        with self.assertRaises(ValueError):verifier.verify(row)

    def test_signal_context_works(self):
        from scripts.reproduce_x3a.train_x3a import stop_requests
        with stop_requests() as state:self.assertIsNone(state['signal'])


    def test_frozen_dataset_uses_relocation_without_changing_id_or_labels(self):
        from unittest.mock import patch
        import torch
        import src.datasets.frozen as frozen
        from src.datasets.media import MediaConfig
        row=self.row();p=Path(row['path'])
        manifest=self.root/'manifest.jsonl'
        manifest.write_text(json.dumps(dict(sample_id=row['sample_id'],video_path=row['source_path'],split='train',
                                           clip_label=1,video_label=0,audio_label=1))+'\n')
        mh=file_sha256(manifest)
        self.elig.write_text(json.dumps(dict(dataset='FAV',sample_id=row['sample_id'],manifest_sha256=mh,
                                            eligible={'audio':True},size=row['size'],mtime_ns=row['mtime_ns']))+'\n')
        registry=self.root/'registry.json'
        registry.write_text(json.dumps({'datasets':{'FAV':{'manifest':'manifest.jsonl','sha256':mh}}}))
        config=self.make_audit(row)
        with patch.object(frozen,'ROOT',self.root), patch('src.utils.local_paths.load_local_paths',return_value=config):
            dataset=frozen.FrozenMediaDataset('FAV','train',MediaConfig(frames=1,crop_size=1),
                                             eligibility_path=self.elig,registry=registry,task='audio')
            with patch.object(frozen,'read_audio',return_value=(torch.ones(16),True)) as reader:
                sample=dataset[0]
        self.assertEqual(sample['sample_id'],'fixed-id')
        self.assertEqual(sample['labels'].tolist(),[1,0,1])
        self.assertEqual(sample['source_path'],'/linux/FAV/a.mp4')
        self.assertEqual(Path(reader.call_args.args[0]),p)


class GPUMonitorTests(unittest.TestCase):
    def test_single_gpu_snapshot_and_thermal_count(self):
        from unittest.mock import patch
        from src.utils.gpu_monitor import snapshot, ThermalCounter, ThermalPolicy
        output='0, GPU-example, RTX 3090, 350, 350, 80, 82, 40, 20, 1024\n'
        with patch('src.utils.gpu_monitor.subprocess.check_output',return_value=output):
            rows=snapshot()
        self.assertEqual(len(rows),1)
        counter=ThermalCounter(ThermalPolicy(78,80,2))
        self.assertFalse(counter.update(rows)['emergency_stop'])
        self.assertTrue(counter.update(rows)['emergency_stop'])


class DurationSummaryTests(unittest.TestCase):
    def test_missing_stream_does_not_silently_change_denominator(self):
        from scripts.windows.analyze_durations import summarize
        result=summarize([{'status':'error','error':'missing audio'}],100)
        self.assertEqual(result['status'],'incomplete_no_mean')
        self.assertNotIn('audio_seconds',result)


class AudioPreprocessingTests(unittest.TestCase):
    def test_real_ffmpeg_resampling_and_right_zero_padding(self):
        import numpy as np
        import soundfile as sf
        import torch
        from src.datasets.media import MediaConfig, read_audio, collate_media
        from src.datasets.transforms import pad_audio
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stereo_8k.wav'
            t=np.arange(8000)/8000
            wave=np.stack([0.1*np.sin(2*np.pi*440*t),0.1*np.sin(2*np.pi*440*t)],axis=1)
            sf.write(path,wave,8000,subtype='PCM_16')
            decoded,present=read_audio(path,MediaConfig())
            self.assertTrue(present)
            self.assertEqual(tuple(decoded.shape),(16000,))
            self.assertEqual(decoded.dtype,torch.float32)
            padded,mask=pad_audio(decoded,284672)
            torch.testing.assert_close(padded[:16000],decoded)
            self.assertEqual(int(torch.count_nonzero(padded[16000:])),0)
            self.assertEqual(int(mask.sum()),16000)
            with self.assertRaisesRegex(ValueError,'Truncation'):
                read_audio(path,MediaConfig(audio_target_num_samples=15999))


class LocalSeedSnapshotTests(unittest.TestCase):
    def test_exact_offline_manifest_verifies_configured_snapshot(self):
        import types
        from unittest.mock import Mock,patch
        from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import download_seed,REPO
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'model.pt').write_bytes(b'weight');(path/'config.yaml').write_text('model: seed')
            manifest=path/'manifest.json';manifest.write_text(json.dumps({'model':{'repo':REPO,'revision':'a'*40,
                'files_sha256':{name:file_sha256(path/name) for name in ('model.pt','config.yaml')}}}))
            hub=types.ModuleType('huggingface_hub');hub.HfApi=Mock(side_effect=AssertionError('No network'))
            hub.snapshot_download=Mock(side_effect=AssertionError('Use the configured local snapshot'))
            with patch.dict('sys.modules',{'huggingface_hub':hub}), patch('src.models.hierarchical_score_fusion_variation.emotion2vec_seed.load_local_paths',return_value={'emotion2vec_snapshot':str(path)}):
                self.assertEqual(download_seed(evaluation_manifest=manifest),(str(path),'a'*40))
                (path/'model.pt').write_bytes(b'tampered')
                with self.assertRaises(ValueError):download_seed(evaluation_manifest=manifest)
