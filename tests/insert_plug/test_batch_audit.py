"""Audit format, media and layout failures using disposable synthetic records."""
from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest
import yaml

from examples.insert_plug.data.dataset import EpisodeWriter
from examples.insert_plug.environment.layouts import manifest_row
from examples.insert_plug.data.recording import RecordingSampler
from examples.insert_plug.data.replay import generate_head_video
from examples.insert_plug.diagnostics.audit_batch import audit_batch, audit_episode, grid_steps
from examples.insert_plug.diagnostics.check_action_replay import load_episode


def make_episode(root, index=0, *, source=(.47, -.05), target=(.43, .09), control_hz=10):
    folder = root / f'pos{index}' / 'episode_000000'
    folder.mkdir(parents=True)
    stem = folder.name
    config = dict(board=dict(source_xy=list(source), target_xy=list(target)),
                  task=dict(control_hz=control_hz), physics=dict(timestep=.001),
                  recording=dict(sample_hz=10), cameras=dict(width=16, height=16, depth=dict(enabled=True)))
    xml = folder / f'{stem}.xml'
    xml.write_text('<mujoco model="synthetic_audit_fixture"/>')
    (folder / f'{stem}.yaml').write_text(yaml.safe_dump(config))
    metrics = dict(success=True, stable_for_final_250ms=True, radial_error_m=0.)
    phases = ['settle_initial', 'approach', 'descend_to_grasp', 'grasp', 'extract',
              'transfer', 'align', 'shallow_insert', 'insertion_check', 'release', 'retract', 'settle_final']
    execution = dict(success=True, failure_reason='', model=xml.name,
                     steps=int(.3 * control_hz), recorded_frames=3,
                     phases=[dict(phase=p) for p in phases], planning=[dict(valid=True)],
                     execution_checks=dict(checks=300, unsafe_contact_count=0, last_violation=None))
    metadata = dict(task='insert_plug', config=config, depth_enabled=True, video_camera='head',
                    video_file=f'{stem}.mp4', model_path=xml.name,
                    model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                    **RecordingSampler(10, .001).metadata(control_hz))
    rgb = np.full((16, 16, 3), [220, 30, 20], np.uint8)
    pose = np.array([.4, 0, .02, 1, 0, 0, 0])
    with EpisodeWriter(folder / f'{stem}.h5', metadata) as writer:
        for time in (0., .1, .2):
            writer.append(dict(timestamp=time, phase='synthetic_fixture',
                images=dict(head=rgb, wrist=rgb[..., ::-1].copy()),
                depth={name: np.full((16, 16), .5, np.float32) for name in ('head', 'wrist')},
                joint_position=np.zeros(7), joint_velocity=np.zeros(7), gripper_width=.02,
                action=np.r_[np.zeros(7), .02], qpos=np.zeros(16), qvel=np.zeros(15), ctrl=np.zeros(8),
                peg_pose=pose, source_pose=np.r_[source, .04, 1, 0, 0, 0],
                target_pose=np.r_[target, .04, 1, 0, 0, 0]))
        writer.finalize(success=True, final_metrics=metrics, execution_result=execution)
    generate_head_video(folder / f'{stem}.h5', folder / f'{stem}.mp4')
    return folder


def update_metadata(folder, change, *, sync_yaml=False):
    path = folder / f'{folder.name}.h5'
    with h5py.File(path, 'r+') as handle:
        metadata = json.loads(handle.attrs['metadata_json'])
        change(metadata)
        handle.attrs['metadata_json'] = json.dumps(metadata)
    if sync_yaml:
        path.with_suffix('.yaml').write_text(yaml.safe_dump(metadata['config']))


def test_batch_audits_real_files_and_ignores_unrelated_root_directories(tmp_path):
    make_episode(tmp_path)
    make_episode(tmp_path, 1, source=(.49, -.07), target=(.45, .11))
    (tmp_path / 'lerobot').mkdir()
    (tmp_path / 'test').mkdir()
    result = audit_batch(tmp_path, positions=2, episodes_per_position=1)
    assert result['valid'], result['errors']
    assert result['episodes'] == 2 and result['total_frames'] == 6
    assert result['groups'][0]['source_steps'] == [1, 1]
    assert result['groups'][1]['target_steps'] == [0, 2]
    assert result['control_hz'] == 10
    assert result['groups'][0]['episodes'][0]['action_hold_s'] == .1


def test_legacy_50hz_collection_requires_explicit_control_rate(tmp_path):
    folder = make_episode(tmp_path, control_hz=50)
    with pytest.raises(ValueError, match='control'):
        audit_episode(folder)
    result = audit_episode(folder, control_hz=50)
    assert result['control_hz'] == 50
    assert result['physics_steps'] == 300
    assert result['action_hold_s'] is None


def test_manifest_selects_control_rate_and_rejects_an_explicit_conflict(tmp_path, monkeypatch):
    make_episode(tmp_path, control_hz=50)
    manifest = tmp_path / 'positions.yaml'
    manifest.write_text(yaml.safe_dump(dict(
        control_hz=50, recording_hz=10, grid_spacing_m=.02,
        base_source_xy=[.45, -.07], base_target_xy=[.45, .07],
        positions=[dict(name='pos0', source_xy=[.47, -.05], target_xy=[.43, .09])],
    )))
    # A synthetic XML has no camera: isolate manifest timing propagation from
    # the separate production camera-containment check.
    import examples.insert_plug.environment.layouts as layouts
    import examples.insert_plug.environment.simulation as simulation
    import mujoco
    from types import SimpleNamespace
    class SyntheticSimulation:
        def __init__(self, *args, **kwargs):
            self.model = None
            self.data = SimpleNamespace(qpos=np.zeros(16), qvel=np.zeros(15), time=0.)
        def close(self):
            pass
    monkeypatch.setattr(simulation, 'Simulation', SyntheticSimulation)
    monkeypatch.setattr(mujoco, 'mj_forward', lambda *args: None)
    monkeypatch.setattr(layouts, 'head_visibility', lambda *args: dict(
        all_inside=True, objects=dict(peg=dict(edge_margin_px=10.))))
    result = audit_batch(tmp_path, positions=1, episodes_per_position=1, manifest=manifest)
    assert result['valid'], result['errors']
    assert result['control_hz'] == 50
    with pytest.raises(ValueError, match='Manifest control'):
        audit_batch(tmp_path, positions=1, episodes_per_position=1,
                    manifest=manifest, control_hz=10)


@pytest.mark.parametrize('field,value', [('control_hz', 50), ('timestep', .002)])
def test_audit_rejects_yaml_rates_that_disagree_with_metadata(tmp_path, field, value):
    folder = make_episode(tmp_path)
    section = 'task' if field == 'control_hz' else 'physics'
    update_metadata(folder, lambda meta: meta['config'][section].update({field: value}), sync_yaml=True)
    with pytest.raises(ValueError, match='YAML'):
        audit_episode(folder)


def test_aligned_actions_require_hold_convention(tmp_path):
    folder = make_episode(tmp_path)
    update_metadata(folder, lambda meta: meta.update(
        action_timing='active_control_target_at_observation; may change before next observation'))
    with pytest.raises(ValueError, match='action convention'):
        audit_episode(folder)
    with pytest.raises(ValueError, match='action convention'):
        load_episode(folder / f'{folder.name}.h5')


@pytest.mark.parametrize('hold', [None, .02, .2])
def test_aligned_actions_require_a_complete_100ms_hold(tmp_path, hold):
    folder = make_episode(tmp_path)
    update_metadata(folder, lambda meta: meta.update(action_hold_duration_s=hold))
    with pytest.raises(ValueError, match='hold duration'):
        audit_episode(folder)
    with pytest.raises(ValueError, match='hold duration'):
        load_episode(folder / f'{folder.name}.h5')


@pytest.mark.parametrize('control_hz,repeats', [(10, 1), (50, 5)])
def test_action_replay_loads_aligned_and_legacy_rates(tmp_path, control_hz, repeats):
    folder = make_episode(tmp_path, control_hz=control_hz)
    source = load_episode(folder / f'{folder.name}.h5')
    assert source['control_steps_per_action'] == repeats
    assert source['action_hold_s'] == .1


def test_action_replay_rejects_execution_duration_mismatch(tmp_path):
    folder = make_episode(tmp_path)
    update_metadata(folder, lambda meta: meta['execution_result'].update(steps=4))
    with pytest.raises(ValueError, match='duration'):
        load_episode(folder / f'{folder.name}.h5')


@pytest.mark.parametrize('limit,valid', [(.4, True), (.3, False), (.2, False)])
def test_batch_duration_limit_is_strict_and_checks_the_complete_rollout(tmp_path, limit, valid):
    make_episode(tmp_path)
    result = audit_batch(tmp_path, positions=1, episodes_per_position=1, max_duration_s=limit)
    assert result['valid'] is valid
    assert result['max_duration_s'] == limit
    if not valid:
        assert 'strictly below' in result['errors'][0]


@pytest.mark.parametrize('limit', [0., -1., float('nan'), float('inf')])
def test_batch_rejects_invalid_duration_limit(tmp_path, limit):
    with pytest.raises(ValueError, match='Duration limit'):
        audit_batch(tmp_path, max_duration_s=limit)


@pytest.mark.parametrize('damage', ['depth', 'video', 'guard', 'sidecar'])
def test_audit_rejects_recordings_with_missing_depth_wrong_video_or_bad_checks(tmp_path, damage):
    folder = make_episode(tmp_path)
    h5 = folder / 'episode_000000.h5'
    if damage == 'depth':
        with h5py.File(h5, 'r+') as handle:
            del handle['observations/depth/wrist']
    elif damage == 'video':
        # Replace video with a wrong camera while retaining valid RGB-D data.
        import imageio.v2 as imageio
        with imageio.get_writer(folder / 'episode_000000.mp4', fps=10, codec='libx264', macro_block_size=1) as writer:
            for _ in range(3):
                writer.append_data(np.full((16, 16, 3), [20, 30, 220], np.uint8))
    elif damage == 'guard':
        with h5py.File(h5, 'r+') as handle:
            metadata = json.loads(handle.attrs['metadata_json'])
            metadata['execution_result']['execution_checks']['checks'] = 299
            handle.attrs['metadata_json'] = json.dumps(metadata)
    else:
        (folder / 'obsolete_result.json').write_text('{}')
    with pytest.raises(ValueError):
        audit_episode(folder)


def test_off_grid_or_reused_holder_positions_fail(tmp_path):
    with pytest.raises(ValueError, match='grid'):
        grid_steps((.451, -.07), (.45, -.07), .02)
    make_episode(tmp_path)
    make_episode(tmp_path, 1)
    result = audit_batch(tmp_path, positions=2, episodes_per_position=1)
    assert not result['valid']
    assert any('reused' in error for error in result['errors'])


def set_collision_mode(folder, enabled):
    def change(meta):
        meta['strict_collision_checks'] = enabled
        meta['config']['task'].setdefault('planning', {})['strict_collision_checks'] = enabled
        meta['execution_result']['execution_checks'].update(
            collision_checks_enabled=enabled, checks=300 if enabled else 0,
            unsafe_contact_count=0 if enabled else None, last_violation=None)
        for plan in meta['execution_result']['planning']:
            plan['collision_checks_enabled'] = enabled
    update_metadata(folder, change, sync_yaml=True)


@pytest.mark.parametrize('enabled', [False, True])
def test_explicit_collision_modes_preserve_success_and_unknown_contact_counts(tmp_path, enabled):
    folder = make_episode(tmp_path)
    set_collision_mode(folder, enabled)
    episode = audit_episode(folder)
    assert episode['collision_checks_enabled'] is enabled
    assert episode['contact_checks'] == (300 if enabled else 0)
    assert episode['unsafe_contacts'] == (0 if enabled else None)
    assert episode['physics_steps'] == 300  # Physical integration remains enabled.
    batch = audit_batch(tmp_path, positions=1, episodes_per_position=1)
    assert batch['valid'], batch['errors']
    assert batch['collision_checked_episodes'] == int(enabled)
    assert batch['collision_unchecked_episodes'] == int(not enabled)
    assert batch['total_contact_checks'] == (300 if enabled else 0)
    assert batch['total_unsafe_contacts'] == (0 if enabled else None)


@pytest.mark.parametrize('damage', [
    'count_zero', 'calls_nonzero', 'violation', 'metadata_mode', 'config_mode',
    'planning_mode', 'missing_mode', 'nonboolean_mode',
])
def test_disabled_collision_audit_rejects_false_or_inconsistent_claims(tmp_path, damage):
    folder = make_episode(tmp_path)
    set_collision_mode(folder, False)

    def change(meta):
        checks = meta['execution_result']['execution_checks']
        if damage == 'count_zero':
            checks['unsafe_contact_count'] = 0
        elif damage == 'calls_nonzero':
            checks['checks'] = 300
        elif damage == 'violation':
            checks['last_violation'] = {'reason': 'contact'}
        elif damage == 'metadata_mode':
            meta['strict_collision_checks'] = True
        elif damage == 'config_mode':
            meta['config']['task']['planning']['strict_collision_checks'] = True
        elif damage == 'planning_mode':
            meta['execution_result']['planning'][0]['collision_checks_enabled'] = True
        elif damage == 'missing_mode':
            del checks['collision_checks_enabled']
        else:
            checks['collision_checks_enabled'] = 0

    update_metadata(folder, change, sync_yaml=True)
    with pytest.raises(ValueError, match='[Cc]ollision|[Cc]ontact|violation|guard'):
        audit_episode(folder)


def test_mixed_contact_monitoring_batch_does_not_turn_unknown_into_zero(tmp_path):
    old = make_episode(tmp_path)
    new = make_episode(tmp_path, 1, source=(.49, -.07), target=(.45, .11))
    set_collision_mode(new, False)
    assert audit_episode(old)['collision_checks_enabled'] is True
    batch = audit_batch(tmp_path, positions=2, episodes_per_position=1)
    assert batch['valid'], batch['errors']
    assert batch['collision_checked_episodes'] == batch['collision_unchecked_episodes'] == 1
    assert batch['total_contact_checks'] == 300
    assert batch['total_unsafe_contacts'] is None


@pytest.mark.parametrize('enabled,nested', [(False, True), (True, True), (False, False), (True, False)])
def test_layout_manifest_preserves_explicit_contact_monitoring_mode(enabled, nested):
    checks = dict(collision_checks_enabled=enabled, unsafe_contact_count=0 if enabled else None)
    checks['checks' if nested else 'contact_checks'] = 300 if enabled else 0
    physics = dict(success=True, duration_seconds=.3, control_steps=3,
                   radial_error_m=0., insertion_depth_m=.03)
    if nested:
        physics['execution_checks'] = checks
    else:
        physics.update(checks)
    row = dict(name='pos0', source_xy=[.47, -.05], target_xy=[.43, .09],
               source_steps=[1, 1], target_steps=[-1, 1], physics_check=physics,
               visibility={phase: dict(all_inside=True, min_edge_margin_px=5.)
                           for phase in ('initial', 'final')})
    compact = manifest_row(row)['physics_check']
    assert compact['collision_checks_enabled'] is enabled
    assert compact['contact_checks'] == (300 if enabled else 0)
    assert compact['unsafe_contact_count'] == (0 if enabled else None)
