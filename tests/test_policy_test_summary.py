"""Independent policy-batch audit contracts using small synthetic artifacts."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('policy_test_summary', ROOT / 'scripts/summarize_insert_plug_policy_test.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def update_group(root, position):
    results = []
    for path in sorted((root / position).glob('episode_*/result.json')):
        result = json.loads(path.read_text())
        results.append(dict(
            **{key: result[key] for key in ('episode_index', 'position', 'success', 'task_success',
                                            'contact_clean', 'stop_reason', 'simulation_seconds')},
            unsafe_contact_count=result['contacts']['unsafe_contact_count'],
            report=f'{path.parent.name}/result.json', video=f'{path.parent.name}/rollout.mp4'))
    checkpoint = json.loads((root / position / 'episode_000000/result.json').read_text())['policy']['checkpoint']
    save(root / position / 'summary.json', dict(
        episodes_requested=10, episodes_completed=len(results), complete=len(results) == 10,
        successes=sum(row['success'] for row in results), checkpoint=checkpoint, results=results))


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / 'evaluation'
    root.mkdir()
    checkpoint = tmp_path / '030000' / 'pretrained_model'
    checkpoint.mkdir(parents=True)
    layouts = yaml.safe_load((ROOT / 'config/insert_plug_positions.yaml').read_text())['positions']
    for i, layout in enumerate(layouts):
        for repetition in range(10):
            folder = root / layout['name'] / f'episode_{repetition:06d}'
            folder.mkdir(parents=True)
            config = dict(task=dict(control_hz=10), physics=dict(timestep=.001),
                          board={key: layout[key] for key in ('source_xy', 'target_xy')},
                          cameras=dict(width=256, height=256))
            result = dict(
                position=layout['name'], repetition=repetition, episode_index=repetition,
                seed=10*i + repetition, config=config,
                policy=dict(checkpoint=str(checkpoint), policy_type='act', n_action_steps=10, calls=2),
                success=False, task_success=False, contact_clean=True, stop_reason='timeout', runtime_error=None,
                control_hz=10., action_hold_s=.1, physics_timestep_s=.001, requested_duration_s=.2,
                completed_control_steps=2, attempted_control_steps=2, selected_actions=2,
                simulation_seconds=.2, wall_seconds=.7, physics_checks=200,
                contacts=dict(unsafe_contact_count=0, checks=200), stop_on_contact=True,
                video='rollout.mp4', video_frames=3, video_layout='head_left_wrist_right', scene='scene.xml',
                final_metrics=dict(success=False), last_action=[0]*8)
            save(folder / 'result.json', result)
            bodies = ''.join(f'<body name="{side}_socket" pos="{layout[side + "_xy"][0]} '
                             f'{layout[side + "_xy"][1]} 0" />' for side in ('source', 'target'))
            (folder / 'scene.xml').write_text(f'<mujoco><option timestep="0.001"/><worldbody>{bodies}</worldbody></mujoco>')
            (folder / 'rollout.mp4').write_bytes(b'fixture-video')
        update_group(root, layout['name'])
    return root, checkpoint


def summarize(batch, **kwargs):
    root, checkpoint = batch
    return audit.summarize_batch(root, checkpoint=checkpoint, video_check='none', duration=.2, **kwargs)


def mutate(batch, change, position='pos0', repetition=0):
    path = batch[0] / position / f'episode_{repetition:06d}' / 'result.json'
    value = json.loads(path.read_text())
    change(value)
    save(path, value)
    return path


def test_all_physical_timeouts_are_complete_and_only_explicit_write_creates_reports(batch):
    root, checkpoint = batch
    summary, rows = summarize(batch)
    assert summary['complete'] and summary['audit_execution_ok'] and not summary['success']
    assert summary['episodes_completed'] == summary['contact_clean'] == summary['timeouts'] == 100
    assert summary['successes'] == summary['task_successes'] == summary['runtime_errors'] == 0
    assert all(group['identical_recorded_outcomes'] for group in summary['positions'])
    assert not summary['all_videos_fully_decoded']
    assert [row['seed'] for row in rows] == list(range(100))
    assert not (root / 'summary.json').exists()
    first = root / rows[0]['report']
    before = first.read_bytes()
    audit.write_reports(root, summary, rows)
    assert first.read_bytes() == before
    with (root / 'episodes.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 100 and csv_rows[-1]['video'] == 'pos9/episode_000009/rollout.mp4'
    assert json.loads((root / 'summary.json').read_text())['complete']
    assert audit.main(['--root', str(root), '--checkpoint', str(checkpoint), '--duration', '.2',
                       '--video-check', 'none']) == 0


@pytest.mark.parametrize('change,match', [
    (lambda value: value.update(seed=1), 'seed must be'),
    (lambda value: value.update(position='pos1'), 'position must be'),
    (lambda value: value.update(repetition=1), 'repetition must be'),
    (lambda value: value['policy'].update(n_action_steps=1), 'n_action_steps'),
    (lambda value: value['policy'].update(checkpoint='/wrong/checkpoint'), 'Checkpoint path'),
    (lambda value: value.update(control_hz=50), 'control_hz mismatch'),
    (lambda value: value['config']['board'].update(source_xy=[0., 0.]), 'vetted fixed position'),
    (lambda value: value.update(video_frames=2), 'every attempted action'),
    (lambda value: value.update(simulation_seconds=.1), 'physics check count'),
])
def test_episode_identity_and_execution_contracts_are_checked(batch, change, match):
    mutate(batch, change)
    summary, rows = summarize(batch)
    assert not summary['complete'] and not summary['audit_valid']
    assert any(match in error for error in summary['errors'])
    assert not rows[0]['audit_valid']


def test_partial_progress_is_explicit_and_missing_report_fails_final_cli(batch):
    root, checkpoint = batch
    (root / 'pos9/episode_000009/result.json').unlink()
    update_group(root, 'pos9')
    summary, _ = summarize(batch)
    assert not summary['complete'] and not summary['reports_complete']
    assert summary['episodes_completed'] == 99
    assert summary['missing_artifacts'] == ['pos9/episode_000009/result.json']
    argv = ['--root', str(root), '--checkpoint', str(checkpoint), '--duration', '.2', '--video-check', 'none']
    assert audit.main(argv) == 1
    assert audit.main([*argv, '--allow-incomplete']) == 0


def test_runtime_error_is_not_timeout_or_artifact_incompleteness(batch):
    mutate(batch, lambda value: value.update(stop_reason='runtime_error',
                                            runtime_error='ValueError: Joint target outside limits'))
    update_group(batch[0], 'pos0')
    summary, rows = summarize(batch)
    assert summary['complete'] and summary['audit_valid'] and summary['infrastructure_ok']
    assert not summary['audit_execution_ok']
    assert summary['timeouts'] == 99 and summary['runtime_errors'] == 1
    assert rows[0]['runtime_error'] == 'ValueError: Joint target outside limits'


def test_contact_stop_is_counted_separately(batch):
    def contact(value):
        value.update(stop_reason='contact_violation', contact_clean=False,
                     runtime_error='ExecutionSafetyError: forbidden contact')
        value['contacts']['unsafe_contact_count'] = 1
    mutate(batch, contact)
    update_group(batch[0], 'pos0')
    summary, _ = summarize(batch)
    assert summary['complete'] and summary['timeouts'] == 99
    assert summary['contact_stops'] == summary['contact_violation_episodes'] == 1
    assert summary['contact_clean'] == 99 and summary['unsafe_contact_count'] == 1


def test_physical_success_requires_stable_successful_final_metrics(batch):
    folder = batch[0] / 'pos0/episode_000000'
    result = json.loads((folder / 'result.json').read_text())
    result.update(success=True, task_success=True, stop_reason='success', requested_duration_s=.3,
                  simulation_seconds=.3, completed_control_steps=3, attempted_control_steps=3,
                  selected_actions=3, physics_checks=300, video_frames=4,
                  continuous_success_seconds=.25, success_required_seconds=.25,
                  final_metrics=dict(success=True))
    result['contacts']['checks'] = 300
    result['policy']['calls'] = 3
    arguments = dict(position='pos0', repetition=0, checkpoint=batch[1],
                     layout=result['config']['board'], video_check='none', duration=.3)
    assert audit.audit_episode(folder, result, **arguments)['video_audit']['check'] == 'none'
    assert audit.totals([result])['successes'] == audit.totals([result])['task_successes'] == 1
    result['continuous_success_seconds'] = .24
    with pytest.raises(ValueError, match='continuous stability'):
        audit.audit_episode(folder, result, **arguments)


@pytest.mark.parametrize('artifact', ['scene.xml', 'rollout.mp4'])
def test_missing_episode_artifacts_cannot_pass(batch, artifact):
    (batch[0] / 'pos0/episode_000000' / artifact).unlink()
    summary, _ = summarize(batch)
    assert not summary['complete'] and not summary['audit_valid']
    assert artifact in summary['errors'][0]


def test_worker_failure_is_retained_even_when_all_reports_exist(batch):
    save(batch[0] / 'launch_summary.json', {'process_results': [{'position': 'pos3', 'returncode': 1}]})
    summary, _ = summarize(batch)
    assert summary['complete'] and summary['audit_valid']
    assert not summary['infrastructure_ok'] and not summary['audit_execution_ok']
    assert summary['infrastructure_failures'] == ['launch_summary.json: pos3 process exit 1']
    assert audit.main(['--root', str(batch[0]), '--checkpoint', str(batch[1]), '--duration', '.2',
                       '--video-check', 'none', '--allow-incomplete']) == 1


def test_actual_launcher_logs_and_outcome_exit_codes_are_accepted(batch):
    root, _ = batch
    process_results = []
    for position in audit.POSITIONS:
        (root / f'{position}.log').write_text('physical outcomes saved\n')
        process_results.append(dict(position=position, returncode=2, complete=True, episodes_completed=10,
                                    command=['python', '-m', 'examples.insert_plug', 'rollout'],
                                    completed_at='2026-11-07T21:00:00+08:00'))
    save(root / 'eval_config.json', dict(started_at='2026-11-07T20:00:00+08:00', process_results=process_results))
    save(root / 'launch_summary.json', dict(complete=True, completed_at='2026-11-07T21:00:00+08:00',
                                           process_results=process_results))
    summary, _ = summarize(batch)
    assert summary['complete'] and summary['infrastructure_ok']
    assert summary['started_at'] == '2026-11-07T20:00:00+08:00'
    assert summary['completed_at'] == '2026-11-07T21:00:00+08:00'


def test_checkpoint_configuration_hash_is_independently_checked(batch):
    (batch[1] / 'config.json').write_text('{}')
    summary, _ = summarize(batch)
    assert not summary['complete']
    assert all('config_sha256 differs' in error for error in summary['errors'])


def test_stale_position_summary_and_changed_initial_scene_fail(batch):
    summary_path = batch[0] / 'pos0/summary.json'
    summary = json.loads(summary_path.read_text())
    summary['successes'] = 9
    save(summary_path, summary)
    scene = batch[0] / 'pos1/episode_000001/scene.xml'
    scene.write_text(scene.read_text().replace('<worldbody>', '<worldbody>\n'))
    report, _ = summarize(batch)
    assert any('summary success count' in error for error in report['errors'])
    assert any('scene_sha256 changed' in error for error in report['errors'])


def test_full_video_decode_checks_actual_count_and_propagates_corruption(tmp_path, monkeypatch):
    import imageio_ffmpeg
    video = tmp_path / 'rollout.mp4'
    video.write_bytes(b'fixture-video')
    monkeypatch.setattr(imageio_ffmpeg, 'read_frames', lambda path: iter_metadata())
    calls = []
    def decode(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr='', stdout='frame=3\nprogress=end\n')
    monkeypatch.setattr(audit.subprocess, 'run', decode)
    result = audit.check_video(video, 3, 512, 256)
    assert result['full_decode'] and result['decoded_frames'] == 3
    assert '-xerror' in calls[0] and 'null' in calls[0]
    with pytest.raises(ValueError, match='frame count'):
        audit.check_video(video, 4, 512, 256)
    monkeypatch.setattr(audit.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stderr='corrupt packet', stdout='frame=1'))
    with pytest.raises(ValueError, match='corrupt packet'):
        audit.check_video(video, 3, 512, 256)


def iter_metadata():
    yield dict(size=(512, 256), fps=10., duration=.3)
