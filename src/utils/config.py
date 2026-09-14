"""Read experiment YAML and reject invalid or unresolved execution settings."""
from pathlib import Path
import hashlib
import json
import yaml


def resolve_sampler_policy(config):
    """Return the execution-level sampler contract independent of YAML formatting."""
    training=config['training']; sampler=training['sampler']; distributed=config.get('distributed',{})
    policy={
        'type': sampler.get('type',sampler.get('name')),
        'replacement': bool(sampler['replacement']),
        'global_epoch_samples': int(sampler.get('global_epoch_samples',14724)),
        'label': sampler['label'],
        'seed': int(config['experiment']['seed']),
        'world_size': int(distributed.get('world_size',1)),
    }
    if policy['type'] is None or policy['global_epoch_samples'] < 1 or policy['world_size'] < 1:
        raise ValueError('Invalid canonical sampler policy')
    return policy


def sampler_policy_sha256(policy):
    payload=json.dumps(policy,sort_keys=True,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def validate_sampler_policy(policy, *, task, expected_world_size=4, expected_samples=14724):
    """Validate semantic sampler settings after canonical resolution."""
    if policy['type'] != 'balanced': raise ValueError('Sampler type must be balanced')
    if policy['replacement'] is not True: raise ValueError('Sampler replacement must be true')
    if policy['global_epoch_samples'] != expected_samples: raise ValueError('Unexpected global epoch sample count')
    if policy['label'] != task+'_label': raise ValueError('Sampler label does not match task')
    if policy['world_size'] != expected_world_size: raise ValueError('Unexpected sampler world size')
    return policy


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiment(path, project_root, *, preflight=False):
    root = Path(project_root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    cfg = yaml.safe_load(path.read_text())
    def require(key):
        value = cfg
        for part in key.split('.'):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f'Missing required setting: {key}')
            value = value[part]
        if value is None:
            raise ValueError(f'Null required setting: {key}')
        return value
    for key in ('experiment.id', 'experiment.stage', 'experiment.seed', 'data.dataset',
                'data.split_manifest', 'data.eligibility', 'data.task.label', 'outputs.report_dir'):
        require(key)
    if require('data.dataset') != 'FAV':
        raise ValueError('This experiment preflight is scoped to FAV')
    if type(require('experiment.seed')) is not int:
        raise ValueError('seed must be an integer')
    if 'fav_in_domain_v1' in require('experiment.id'):
        raise ValueError('Use fav_indomain_v1')
    if require('data.split_manifest') != 'configs/fixed_manifests.json':
        raise ValueError('Use the frozen manifest registry')
    if require('data.eligibility') != 'configs/phase0_data_quality_v1.json':
        raise ValueError('Use the frozen quality registry')
    for key in ('data.split_manifest', 'data.eligibility'):
        if not (root / require(key)).is_file():
            raise ValueError(f'Missing registry: {key}')
    blockers = []
    if 'distributed' in cfg:
        d=cfg['distributed']; t=cfg['training']
        valid_recipes=(
            {'backend':'nccl','world_size':4,'devices':[0,1,2,3],'find_unused_parameters':False},
            {'backend':'nccl','world_size':3,'devices':[1,2,3],'find_unused_parameters':False},
            {'backend':'nccl','world_size':2,'devices':[1,2],'find_unused_parameters':False},
        )
        if d not in valid_recipes:
            raise ValueError('Expected explicit three- or four-GPU NCCL recipe')
        if t['batch_size'] != t['per_device_batch_size'] or t['global_effective_batch_size'] != d['world_size']*t['per_device_batch_size']*t['gradient_accumulation_steps']:
            raise ValueError('Distributed batch accounting mismatch')
    if 'model' in cfg:
        name = require('model.name')
        if name not in ('x3d_m', 'x3d_s', 'x3d_xs', 'aasist'):
            raise ValueError('Unsupported model')
        for key in ('training.batch_size', 'training.epochs', 'training.gradient_accumulation_steps'):
            if type(require(key)) is not int or require(key) <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if require('training.optimizer.learning_rate') <= 0:
            raise ValueError('learning_rate must be positive')
        for key in ('training.optimizer.name', 'training.loss.name', 'training.scheduler.name',
                    'checkpoint_selection.metric', 'checkpoint_selection.mode', 'outputs.checkpoint_dir'):
            require(key)
        if require('data.split') != {'train': 'train', 'selection': 'dev', 'final_evaluation': 'test'}:
            raise ValueError('Invalid fixed split roles')
        resolve_sampler_policy(cfg)
        if require('checkpoint_selection.split') != 'dev':
            raise ValueError('Checkpoint selection must use dev')
        if name == 'aasist':
            if require('model.class_mapping') != {'real': 0, 'fake': 1} or require('model.fake_class_index') != 1:
                raise ValueError('Fresh AASIST mapping must be real=0, fake=1')
            if cfg['model'].get('pretrained_checkpoint') is not None:
                raise ValueError('This recipe is fresh AASIST training')
            if require('data.task.label') != 'audio_label' or require('training.loss.name') != 'cross_entropy':
                raise ValueError('AASIST requires audio labels and CE')
            for key in ('first_conv', 'filts', 'gat_dims', 'pool_ratios', 'temperatures'):
                require('model.official_config.' + key)
            if require('audio.sample_rate') != 16000 or require('audio.channels') != 'mono':
                raise ValueError('Expected 16 kHz mono')
            if require('audio.truncation') != 'prohibited' or require('audio.padding') != 'right_zero_pad':
                raise ValueError('Unexpected audio policy')
            policy = require('audio.length_policy')
            if policy == 'unresolved_preflight_required':
                blockers.append('audio.length_policy/target_num_samples unresolved')
            elif policy == 'pad_to_eligible_train_max':
                if type(require('audio.target_num_samples')) is not int or require('audio.target_num_samples') <= 0:
                    raise ValueError('Audio target must be a positive integer')
            else:
                raise ValueError('No executable audio length policy has been approved yet')
            lock = json.loads((root / 'configs/third_party.lock.json').read_text())['aasist']
            if require('third_party.commit') != lock['commit'] or require('third_party.path') != lock['path']:
                raise ValueError('AASIST vendor lock mismatch')
        else:
            if require('data.task.label') != 'video_label' or require('training.loss.name') != 'bce_with_logits':
                raise ValueError('X3D requires video labels and BCE logits')
            if require('video.num_frames') != 128 or require('video.sampling') != 'uniform_over_full_video':
                raise ValueError('Expected full-video 128-frame sampling')
            if not 0 < require('video.crop_size') <= require('video.resize_short_side'):
                raise ValueError('Invalid video spatial sizes')
            for key in ('mean', 'std'):
                values = require('video.normalization.' + key)
                if len(values) != 3 or (key == 'std' and any(x <= 0 for x in values)):
                    raise ValueError('Invalid RGB normalization')
    else:
        if require('data.task.label') != 'clip_label' or require('fusion.name') != 'probabilistic_or' or require('fusion.trainable') is not False:
            raise ValueError('Invalid OR baseline')
        if require('inputs.aasist_fake_class_index') != 1:
            raise ValueError('AASIST fake index must be 1')
        for model in ('x3d', 'aasist'):
            if not (root / require(f'inputs.{model}_config')).is_file():
                raise ValueError('Missing modality config')
            checkpoint = cfg['inputs'].get(model + '_checkpoint')
            if checkpoint is None:
                blockers.append(model + ' checkpoint unresolved')
            elif not (root / checkpoint).is_file():
                raise ValueError('Missing checkpoint')
        if not 0 <= require('evaluation.fixed_threshold') <= 1:
            raise ValueError('Invalid threshold')
    if blockers and not preflight:
        raise ValueError('; '.join(blockers))
    return cfg, blockers
