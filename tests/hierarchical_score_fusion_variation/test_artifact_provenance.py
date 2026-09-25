import copy
import unittest
from src.models.hierarchical_score_fusion_variation.artifact_provenance import RELOCATIONS,canonical_artifact_recipe,artifact_recipe_digest

class ProvenanceTests(unittest.TestCase):
    def recipes(self):
        old={'implementation':{p:v[1] for p,v in RELOCATIONS.items()},'weights':'same','video_recipe':{'frames':128}}
        new={'implementation':{v[0]:v[2] for v in RELOCATIONS.values()},'weights':'same','video_recipe':{'frames':128}}
        return old,new
    def test_relocation_only(self):
        old,new=self.recipes();self.assertEqual(artifact_recipe_digest(old),artifact_recipe_digest(new))
    def test_not_mutated(self):
        old,_=self.recipes();original=copy.deepcopy(old);canonical_artifact_recipe(old);self.assertEqual(original,old)
    def test_weights_differ(self):
        old,new=self.recipes();new['weights']='changed';self.assertNotEqual(artifact_recipe_digest(old),artifact_recipe_digest(new))
    def test_new_code_differ(self):
        old,new=self.recipes();new['implementation'][next(iter(new['implementation']))]='unknown';self.assertNotEqual(artifact_recipe_digest(old),artifact_recipe_digest(new))
    def test_unreviewed_legacy(self):
        old,_=self.recipes();old['implementation'][next(iter(old['implementation']))]='unknown'
        with self.assertRaises(ValueError):canonical_artifact_recipe(old)
    def test_preprocessing_differ(self):
        old,new=self.recipes();new['video_recipe']['frames']=16;self.assertNotEqual(artifact_recipe_digest(old),artifact_recipe_digest(new))
    def test_collision(self):
        old,_=self.recipes();p=next(iter(RELOCATIONS));v=RELOCATIONS[p];old['implementation'][v[0]]=v[2]
        with self.assertRaises(ValueError):canonical_artifact_recipe(old)
if __name__=='__main__':unittest.main()
