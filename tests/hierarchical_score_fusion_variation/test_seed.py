import unittest
import numpy as np
import torch
from src.models.hierarchical_score_fusion_variation.emotion2vec_seed import CLASSES,ordered_probabilities
from src.datasets.hierarchical_score_fusion_variation.extract_trajectories import auditory_seed
from src.datasets.hierarchical_score_fusion_variation.variation_features import change_features
from src.branches.hierarchical_score_fusion_variation.hierarchical_emotion_variation_branch import HierarchicalEmotionVariationBranch

class SeedTests(unittest.TestCase):
    def test_labels(self):
        p=ordered_probabilities(dict(labels=['中文/'+x for x in CLASSES[::-1]],scores=[1]+[0]*8))
        self.assertEqual(p[-1],1)
    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):ordered_probabilities(dict(labels=CLASSES,scores=[1]*9))
    def test_unknown_retained_in_variation(self):
        r=change_features(np.eye(9)[[0,8]],[1.5,2.5],[True,True],kind='audio')
        self.assertEqual(r['features'].shape,(30,));self.assertEqual(r['transitions'][0,8],1)
    def test_no_cross_invalid(self):
        r=change_features(np.eye(9)[:3],[1.5,2.5,3.5],[True,False,True],kind='audio')
        self.assertEqual(r['valid_pairs'],0)
    def test_complete_windows(self):
        class Dummy:
            def probabilities(self,x):return np.ones(9,dtype=np.float32)/9
        a=auditory_seed(np.ones(64000),Dummy())
        self.assertEqual(a['values'].shape,(2,9));np.testing.assert_allclose(a['times'],[1.5,2.5])
        self.assertEqual(auditory_seed(np.ones(47000),Dummy())['values'].shape,(0,9))
    def test_silence_invalid(self):
        class Dummy:
            def probabilities(self,x):raise AssertionError('Silence must be excluded')
        a=auditory_seed(np.zeros(64000),Dummy());self.assertFalse(a['valid'].any())
    def test_head_and_restore(self):
        b=HierarchicalEmotionVariationBranch(audio_dim=30)
        v=torch.randn(10,27);a=torch.randn(10,30)
        b.video_head.fit_standardizer(v);b.audio_head.fit_standardizer(a)
        out=b(v,a);(out['video_logits'].sum()+out['audio_logits'].sum()).backward()
        self.assertTrue(all(p.grad is not None for p in b.parameters()))
        c=HierarchicalEmotionVariationBranch(audio_dim=30);c.load_state_dict(b.state_dict())
        b.eval();c.eval();torch.testing.assert_close(b(v,a)['score_emotion'],c(v,a)['score_emotion'])
    def test_old_head_rejected(self):
        with self.assertRaises(RuntimeError):
            HierarchicalEmotionVariationBranch(audio_dim=30).load_state_dict(HierarchicalEmotionVariationBranch().state_dict())
if __name__=='__main__':unittest.main()
