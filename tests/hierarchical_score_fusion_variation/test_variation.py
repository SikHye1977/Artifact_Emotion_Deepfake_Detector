import unittest
import numpy as np
import torch
from src.datasets.hierarchical_score_fusion_variation.variation_features import change_features
from src.branches.hierarchical_score_fusion_variation.hierarchical_emotion_variation_branch import HierarchicalEmotionVariationBranch

class VariationTests(unittest.TestCase):
    def test_video_rate(self):
        x=np.eye(8)[[0,1]]
        r=change_features(x,[0,.5],[True,True],kind='video')
        self.assertEqual(r['features'].shape,(27,))
        np.testing.assert_allclose(r['transitions'][0,:2],[-2,2])
        self.assertEqual(r['transitions'][0,-1],2)
    def test_invalid_not_bridged(self):
        r=change_features(np.eye(8)[[0,1,2]],[0,.5,1],[True,False,True],kind='video')
        self.assertEqual(r['valid_pairs'],0)
    def test_track_and_gap(self):
        for times,segments in [([0,.5],[0,1]),([0,2],[0,0])]:
            r=change_features(np.eye(8)[[0,1]],times,[True,True],kind='video',segments=segments)
            self.assertEqual(r['valid_pairs'],0)
    def test_constant(self):
        r=change_features(np.ones((3,8))/8,[0,.5,1],[True]*3,kind='video')
        np.testing.assert_array_equal(r['features'],np.zeros(27))
        self.assertEqual(r['valid_pairs'],2)
    def test_audio_video_identical_formula(self):
        x=np.eye(8)[:2]
        a=change_features(x,[0,1],[True]*2,kind='audio')
        b=change_features(x,[0,1],[True]*2,kind='video')
        self.assertEqual(a['features'].shape,(27,))
        np.testing.assert_allclose(a['features'],b['features'])
        self.assertEqual(a['transitions'][0,-1],1)
    def test_embedding_rejected(self):
        with self.assertRaises(ValueError):
            change_features(np.ones((2,256)),[0,1],[True]*2,kind='audio')
    def test_empty_audio(self):
        r=change_features(np.empty((0,8)),[],[],kind='audio')
        self.assertEqual(r['valid_pairs'],0)
        self.assertTrue(np.isfinite(r['features']).all())
    def test_train_heads(self):
        b=HierarchicalEmotionVariationBranch()
        v=torch.randn(6,27);a=torch.randn(6,27)
        b.video_head.fit_standardizer(v);b.audio_head.fit_standardizer(a)
        out=b(v,a);loss=out['video_logits'].square().mean()+out['audio_logits'].square().mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters()))
        self.assertEqual(out['score_emotion'].shape,(6,))
    def test_timestamp_rejected(self):
        with self.assertRaises(ValueError):change_features(np.eye(8)[:2],[1,1],[True]*2,kind='video')

if __name__=='__main__':unittest.main()
