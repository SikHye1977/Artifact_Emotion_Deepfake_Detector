"""Recognize only the source relocation reviewed on 2026-09-22.

Original manifests/hashes stay untouched. All non-relocation fields remain strict.
"""
import copy
import hashlib
import json

# old path -> (new path, exact audited old/new file hashes)
RELOCATIONS = {
 'scripts/hierarchical_score_fusion_script.py': ('scripts/hierarchical_score_fusion/hierarchical_score_fusion_scripts.py', '4ee0e02ee5317d39a71064906597f9468ce274a7e94fe56122746398511d76c2', '56770072e97b5cc17cf14db43bbd0a2ccf4f900eac8f6b61bda24ec3026e6f9c'),
 'src/branches/hierarchical_artifact_branch.py': ('src/branches/hierarchical_score_fusion/hierarchical_artifact_branch.py', '4d6f54b2ca154985cb7b250a4f40604363fed3698722e1b4d7e7eb90aa9c7f64', '7c29eabb279d42456a1f960fe4e1866e9faafabfb2faf1cbbd422c3647e9d606'),
 'src/branches/hierarchical_emotion_branch.py': ('src/branches/hierarchical_score_fusion/hierarchical_emotion_branch.py', '8009cdbfc3ff3008448178ad79d0afdefde6b95fcc86e24d38c97b540650d9f4', 'c54a20e636475226211b89cc90de56b09acaf5406acef7aae9448717da57eab9'),
 'src/datasets/hierarchical_emotion_inputs.py': ('src/datasets/hierarchical_score_fusion/hierarchical_emotion_inputs.py', 'd40f42b65df05d90ea7830ca817fa165eb5d156a4137d70bba85d61698562594', 'd40f42b65df05d90ea7830ca817fa165eb5d156a4137d70bba85d61698562594'),
 'src/fusion/hierarchical_score_fusion.py': ('src/fusion/hierarchical_score_fusion/hierarchical_score_fusion.py', '56ab212675f9742fc2c7b8759ae16e5d873bc176a44817f6576482b46b296df5', '56ab212675f9742fc2c7b8759ae16e5d873bc176a44817f6576482b46b296df5'),
 'src/models/emotion_deepfake_head.py': ('src/models/hierarchical_score_fusion/emotion_deepfake_head.py', '5ebee50e576ef467d3075bf7bdde046b761a8bf5d560e3590585e67a93da01a2', '5ebee50e576ef467d3075bf7bdde046b761a8bf5d560e3590585e67a93da01a2'),
 'src/models/hierarchical_checkpoint_loading.py': ('src/models/hierarchical_score_fusion/hierarchical_checkpoint_loading.py', '88b6ba8dc46c187918bda8590c2043f34cd5f1eb009ee6ff61e8aaa1de194b75', '88b6ba8dc46c187918bda8590c2043f34cd5f1eb009ee6ff61e8aaa1de194b75'),
}

def canonical_artifact_recipe(recipe):
    result=copy.deepcopy(recipe)
    implementation=result.get('implementation',{})
    for old,(new,old_hash,new_hash) in RELOCATIONS.items():
        if old not in implementation:continue
        if implementation[old]!=old_hash:
            raise ValueError(f'Unreviewed legacy source hash: {old}')
        if new in implementation:
            raise ValueError(f'Ambiguous duplicate source paths: {old}, {new}')
        del implementation[old]
        implementation[new]=new_hash
    return result

def artifact_recipe_digest(recipe):
    return hashlib.sha256(json.dumps(canonical_artifact_recipe(recipe),sort_keys=True,separators=(',',':')).encode()).hexdigest()
