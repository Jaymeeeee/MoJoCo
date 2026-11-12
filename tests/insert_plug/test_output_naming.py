"""Diagnostic names reflect model inputs and actual layouts, without model loading."""
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from examples.insert_plug.policy import output_naming as naming


@pytest.fixture
def naming_inputs(tmp_path, monkeypatch):
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    train = [dict(source_xy=[.41, -.07], target_xy=[.49, .07]),
             dict(source_xy=[.43, -.05], target_xy=[.47, .09])]
    heldout = [dict(source_xy=[.42, -.06], target_xy=[.48, .08])]
    for filename, layouts in [('insert_plug_positions.yaml', train),
                              ('insert_plug_generalization_v1.yaml', heldout)]:
        (config_dir / filename).write_text(yaml.safe_dump(dict(positions=layouts)))
    monkeypatch.setattr(naming, 'CONFIG_DIR', config_dir)
    return dict(parent=tmp_path / 'attention',
                checkpoint=tmp_path / 'outputs/train/act_insert_plug_v1_rgb_20261108_154950/artifacts/checkpoints/030000/pretrained_model',
                saved=dict(type='act', input_features={'observation.images.head': {}}),
                config=dict(board=train[0]), kind='chunked', started_at='2026-11-12T23:30:15+00:00')


@pytest.mark.parametrize('model,depth,kind', [
    ('act', False, 'chunked'), ('act', True, 'ensemble'),
    ('diffusion', False, None), ('diffusion', True, None),
])
def test_names_use_saved_model_inputs_and_shanghai_start(naming_inputs, model, depth, kind):
    args = naming_inputs
    args['saved']['type'] = model
    if depth:
        args['saved']['depth_camera_map'] = {'observation.images.head': 'observation.depth.head'}
    args['kind'] = kind
    # The timestamp belongs to this diagnostic, not the checkpoint's training run.
    result = naming.experiment_output_path(None, **args)
    experiment = f'_{kind}' if kind else ''
    modality = 'rgbd' if depth else 'rgb'
    assert result == args['parent'] / f'{model}_insert_plug_v1_{modality}_ind{experiment}_20261113_073015'
    assert not result.exists()


def test_heldout_label_uses_actual_pair_not_position_name(naming_inputs):
    naming_inputs['config'] = dict(position='pos0', board=dict(source_xy=[.42, -.06], target_xy=[.48, .08]))
    naming_inputs['kind'] = 'stateless'
    assert naming.experiment_output_path(None, **naming_inputs).name == 'act_insert_plug_v1_rgb_ood_stateless_20261113_073015'


def test_unknown_pair_does_not_get_an_ind_or_ood_label(naming_inputs):
    # Each point is seen separately, but this pair is absent from the training layouts.
    naming_inputs['config']['board'] = dict(source_xy=[.41, -.07], target_xy=[.47, .09])
    assert naming.experiment_output_path(None, **naming_inputs).name == 'act_insert_plug_v1_rgb_chunked_20261113_073015'


def test_missing_layout_is_not_guessed(naming_inputs):
    naming_inputs['config'] = {}
    assert naming.experiment_output_path(None, **naming_inputs).name == 'act_insert_plug_v1_rgb_chunked_20261113_073015'


@pytest.mark.parametrize('field,replacement,reason', [
    ('checkpoint', Path('/tmp/pretrained_model'), 'version'),
    ('saved', dict(type='act', input_features={}), 'RGB/RGB-D'),
    ('saved', dict(type='unknown', input_features={'observation.images.head': {}}), 'model type'),
])
def test_unknown_identity_requires_explicit_output(naming_inputs, field, replacement, reason):
    naming_inputs[field] = replacement
    with pytest.raises(ValueError, match=reason):
        naming.experiment_output_path(None, **naming_inputs)


def test_explicit_output_bypasses_identity_and_layout_inference(tmp_path):
    output = tmp_path / 'custom-result'
    assert naming.experiment_output_path(
        output, parent=None, checkpoint=None, saved=None, config=None,
        kind=None, started_at=None) == output
    assert not output.exists()


def test_version_is_read_from_checkpoint_training_directory(naming_inputs):
    naming_inputs['checkpoint'] = Path('/saved/diffusion_insert_plug_v12_rgbd_20261112_110000/artifacts/checkpoints/last/pretrained_model')
    assert naming.experiment_output_path(None, **naming_inputs).name.startswith('act_insert_plug_v12_rgb_ind_')


def test_timestamp_requires_timezone(naming_inputs):
    naming_inputs['started_at'] = '2026-11-12T23:30:15'
    with pytest.raises(ValueError, match='timezone'):
        naming.experiment_output_path(None, **naming_inputs)
    created = datetime.fromisoformat(naming.experiment_timestamp())
    assert created.utcoffset().total_seconds() == 8 * 3600
