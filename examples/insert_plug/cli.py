"""Command-line scene building, expert collection, inspection and replay."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from mujoco_tools.paths import WORKSPACE_ROOT, EXAMPLES_DIR, OUTPUTS_DIR, DATA_DIR, CHECKS_DIR


ROOT = WORKSPACE_ROOT
DEFAULT_SCENE_PATH = EXAMPLES_DIR / 'insert_plug' / 'scene.xml'


def _default_output(args):
    if args.no_images:
        parent = CHECKS_DIR / 'insert_plug'
    else:
        parent = DATA_DIR / 'insert_plug'
    return parent / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


def _configure_gl(backend):
    if backend == 'auto':
        # Respect an explicit user choice. Without a display use a portable
        # baseline; doctor can determine whether EGL is available on this host.
        backend = os.environ.get('MUJOCO_GL', 'glfw' if os.environ.get('DISPLAY') else 'osmesa')
    os.environ['MUJOCO_GL'] = backend
    if backend == 'osmesa':
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    elif backend == 'egl':
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
    else:
        os.environ.pop('PYOPENGL_PLATFORM', None)
    return backend


def view_scene(args):
    """Inspect the initial scene without starting the expert or data writer."""
    _configure_gl('glfw')
    from tempfile import TemporaryDirectory
    from .scene import load_config, build_scene
    from .simulation import Simulation
    from mujoco_tools.viewer import view_static
    from mujoco_tools.cameras import camera_id

    cfg = load_config(args.config)
    with TemporaryDirectory(prefix='mujoco-scene-view-') as folder:
        scene_path = build_scene(cfg, Path(folder) / 'scene.xml')
        sim = Simulation(scene_path, cfg, render=False)
        try:
            print('Scene preview: static scene, no expert and no data recording. Close the window to exit.', flush=True)
            base = cfg['robot']['base_position']
            board = cfg['board']['centre_xy']
            lookat = [(base[0] + board[0]) / 2, (base[1] + board[1]) / 2,
                      cfg['table']['surface_z'] + .25]
            camera = args.camera
            if camera != 'free':
                camera = sim.model.camera(camera_id(sim.model, camera)).name
            view_static(sim.model, sim.data, camera=camera, lookat=lookat)
        finally:
            sim.close()
    return 0


def snapshot_scene(args):
    """Save static initial-state pictures without starting collection."""
    backend = _configure_gl(args.backend)
    from .scene import load_config
    from .snapshot import render_snapshot

    result = render_snapshot(load_config(args.config), args.output, width=args.width)
    result['backend'] = backend
    _print(result)
    return 0


def collect(args):
    backend = _configure_gl(args.backend)
    import numpy as np
    import yaml
    import mujoco
    from .scene import load_config, build_scene
    from .simulation import Simulation
    from .expert import Expert
    from .dataset import EpisodeWriter, validate_episode

    cfg = load_config(args.config)
    if args.width:
        cfg['cameras']['width'] = args.width
        cfg['cameras']['height'] = args.width
    if args.control_hz:
        cfg['task']['control_hz'] = args.control_hz
    if args.sample_hz is not None:
        cfg.setdefault('recording', {})['sample_hz'] = args.sample_hz
    output = Path(args.output).expanduser().resolve() if args.output else _default_output(args)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i in range(args.episodes):
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        episode_cfg = copy.deepcopy(cfg)
        for key in ('source_xy', 'target_xy'):
            episode_cfg['board'][key] = (np.array(cfg['board'][key])
                + rng.uniform(-args.randomize_mm, args.randomize_mm, 2) / 1000).tolist()
        stem = f'episode_{seed:06d}'
        episode_output = output / stem
        if episode_output.exists() or any((output / f'{stem}{suffix}').exists()
               for suffix in ('.h5', '.h5.partial', '.yaml', '.xml', '.mp4', '.partial.mp4')):
            raise FileExistsError(f'Episode {stem} already exists; choose another output directory or seed')
        episode_output.mkdir()
        scene_path = build_scene(episode_cfg, episode_output / f'{stem}.xml')
        (episode_output / f'{stem}.yaml').write_text(yaml.safe_dump(episode_cfg, sort_keys=False), encoding='utf8')
        sim = Simulation(scene_path, episode_cfg, render=not args.no_images)
        writer = None
        result = None
        try:
            expert = Expert(sim, realtime=args.realtime, verbose=not args.quiet)
            expert.prepare()
            if args.viewer:
                import mujoco.viewer
                sim.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            if not args.no_images:
                metadata = dict(seed=seed, config=episode_cfg, task='insert_plug',
                    model_sha256=hashlib.sha256(Path(scene_path).read_bytes()).hexdigest(),
                    model_path=scene_path.name, mujoco_version=mujoco.__version__, renderer_backend=backend,
                    expert='Collision-checked quintic Cartesian plans with measured-grasp suffix and execution gates',
                    robot_gravity_compensation=episode_cfg['robot']['gravity_compensation'],
                    geometry_status=('user STL visuals and decomposed collisions; SI scale inferred from mm; '
                                     'mass and friction are configurable assumptions'
                                     if episode_cfg['peg'].get('type') == 'stl_bolt'
                                     else 'parameterized proxy'),
                    policy_inputs=['observation.images.head', 'observation.images.wrist', 'observation.state'],
                    depth_enabled=sim.depth_enabled,
                    depth_features=(['observation.depth.head', 'observation.depth.wrist'] if sim.depth_enabled else []),
                    camera_calibration=sim.camera_calibration())
                metadata.update(expert.recording_sampler.metadata(sim.control_hz))
                metadata.update(video_camera='head', video_file=f'{stem}.mp4')
                asset_dir = Path(episode_cfg['robot']['model_dir'])
                if not asset_dir.is_absolute():
                    asset_dir = ROOT / asset_dir
                manifest = asset_dir / 'SOURCE.json'
                if manifest.is_file():
                    metadata['robot_asset_manifest'] = json.loads(manifest.read_text())
                gripper_dir = episode_cfg['gripper'].get('model_dir')
                if gripper_dir:
                    gripper_manifest = ROOT / gripper_dir / 'SOURCE.json'
                    if gripper_manifest.is_file():
                        metadata['gripper_asset_manifest'] = json.loads(gripper_manifest.read_text())
                metadata['object_assets'] = {}
                for name in ('board', 'peg'):
                    for key in ('visual_mesh', 'collision_manifest'):
                        value = episode_cfg[name].get(key)
                        if value:
                            asset = ROOT / value
                            metadata['object_assets'][f'{name}/{key}'] = {
                                'path': str(asset), 'sha256': hashlib.sha256(asset.read_bytes()).hexdigest()}
                writer = EpisodeWriter(episode_output / f'{stem}.h5', metadata)
            expert.writer = writer
            result = expert.run()
            result.update(seed=seed, backend=backend, model=scene_path.name)
            result['recorded_frames'] = writer.steps if writer is not None else 0
            if writer is not None:
                path = writer.finalize(success=result['success'], failure_reason=result['failure_reason'],
                    final_metrics=result['metrics'], execution_result=copy.deepcopy(result))
                result['dataset'] = str(path)
                result['validation'] = validate_episode(path)
                if not result['validation']['valid']:
                    result['success'] = False
                    result['failure_reason'] = 'dataset_validation_failed'
                else:
                    from .replay import generate_head_video
                    result['video'] = str(generate_head_video(path, episode_output / f'{stem}.mp4'))
            _print({k: result[k] for k in ('seed', 'success', 'failure_reason', 'steps', 'recorded_frames', 'metrics')})
        except Exception as exc:
            result = dict(seed=seed, success=False, error=f'{type(exc).__name__}: {exc}')
            _print(result)
            raise
        finally:
            if writer is not None:
                writer.close()
            sim.close()
        summaries.append(result)
    summary = dict(episodes=len(summaries), successes=sum(x['success'] for x in summaries),
        success_rate=sum(x['success'] for x in summaries)/len(summaries),
        results=summaries)
    if args.video and not args.no_images:
        from .replay import generate_preview
        video_path = OUTPUTS_DIR / 'insert_plug' / 'previews' / output.name / 'preview.mp4'
        generate_preview(summaries[0]['dataset'], video_path)
        print(f'Video: {video_path}', flush=True)
    print(f'Output: {output}; successes {summary["successes"]}/{summary["episodes"]}', flush=True)
    return 0 if summary['successes'] == summary['episodes'] else 2


def plan_scene(args):
    """Export the complete nominal task without stepping or recording images."""
    from tempfile import TemporaryDirectory
    from .scene import load_config, build_scene
    from .simulation import Simulation
    from .planning import TaskPlanner
    cfg = load_config(args.config)
    with TemporaryDirectory(prefix='insert-plug-plan-') as folder:
        sim = Simulation(build_scene(cfg, Path(folder)/'scene.xml'), cfg, render=False)
        try:
            plan = TaskPlanner(sim).plan()
            files = plan.save(args.output, stem='nominal_trajectory')
            _print(dict(files=files, stages=[s['phase'] for s in plan.stages],
                        reference_duration_seconds=float(plan.time[-1]), validation=plan.validation))
        finally:
            sim.close()
    return 0


def rollout_policy(args):
    """Keep inference dependencies optional for all existing scene commands."""
    import yaml

    policy_config = Path(args.policy_config).expanduser()
    if not policy_config.is_absolute():
        policy_config = ROOT / policy_config
    settings = yaml.safe_load(policy_config.read_text(encoding='utf-8')) or {}
    allowed = {'checkpoint', 'scene_config', 'positions_config', 'device',
               'n_action_steps', 'duration', 'position'}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError(f'Invalid policy configuration in {policy_config}; allowed keys: {sorted(allowed)}')

    def choose(name, default=None):
        value = getattr(args, name, None)
        return settings.get(name, default) if value is None else value

    def workspace_path(value):
        if value is None:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    checkpoint = choose('checkpoint')
    if not checkpoint:
        raise ValueError('Specify a checkpoint in the policy YAML or with --checkpoint')
    position = choose('position', 'config')
    position = None if position is None or str(position) == 'config' else str(position)
    backend = _configure_gl('glfw' if args.viewer else args.backend)
    from .rollout import run_rollouts

    result = run_rollouts(
        checkpoint=workspace_path(checkpoint),
        config=workspace_path(args.config or settings.get('scene_config')),
        positions_config=workspace_path(choose('positions_config')),
        output=args.output, position=position, episodes=args.episodes, seed=args.seed,
        duration=choose('duration', 60.), device=choose('device', 'auto'),
        n_action_steps=choose('n_action_steps', 1), viewer=args.viewer,
        realtime=args.realtime, video=args.video,
        stop_on_contact=not args.continue_on_contact, quiet=args.quiet,
    )
    _print(dict(backend=backend, **result))
    return 0 if result['success'] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    rollout = sub.add_parser('rollout', help='Run a trained LeRobot ACT policy in closed-loop simulation')
    rollout.add_argument('--policy-config', default=str(ROOT / 'config' / 'insert_plug_policy.yaml'))
    rollout.add_argument('--checkpoint', help='Checkpoint step directory or its pretrained_model directory')
    rollout.add_argument('--config', help='Scene YAML (overrides scene_config in policy YAML)')
    rollout.add_argument('--positions-config', help='Fixed layout manifest YAML')
    rollout.add_argument('--position', choices=['config', 'all'] + [str(i) for i in range(10)],
                         help='Use scene-config positions, one saved layout, or all ten layouts')
    rollout.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    rollout.add_argument('--n-action-steps', type=int, help='Actions executed per ACT prediction (defaults to policy YAML)')
    rollout.add_argument('--duration', type=float, help='Maximum simulated seconds per episode (default: 60)')
    rollout.add_argument('--episodes', type=int, default=1, help='Episodes per selected layout')
    rollout.add_argument('--seed', type=int, default=0)
    rollout.add_argument('--output', help='New evaluation directory; existing directories are never overwritten')
    rollout.add_argument('--backend', choices=['auto', 'osmesa', 'egl', 'glfw'], default='auto')
    rollout.add_argument('--viewer', action='store_true', help='Open a live window; requires a graphical desktop')
    rollout.add_argument('--realtime', action='store_true', help='Pace simulation to wall time when compute permits')
    rollout.add_argument('--video', action=argparse.BooleanOptionalAction, default=True,
                         help='Save a side-by-side head/wrist MP4 (default: enabled)')
    rollout.add_argument('--continue-on-contact', action='store_true',
                         help='Diagnostic mode: continue after unsafe contact, but keep episode unsuccessful')
    rollout.add_argument('--quiet', action='store_true')
    plan = sub.add_parser('plan', help='Validate and save the complete nominal trajectory without executing it')
    plan.add_argument('--config', default=None)
    plan.add_argument('--output', default=str(OUTPUTS_DIR/'insert_plug'/'plans'))
    view = sub.add_parser('view', help='Inspect the static scene without motion or data collection')
    view.add_argument('--config', default=None)
    view.add_argument('--camera', choices=['free', 'head', 'third_person', 'wrist'], default='free')
    snapshot = sub.add_parser('snapshot', help='Save pictures of the static initial scene without motion or data collection')
    snapshot.add_argument('--config', default=None)
    snapshot.add_argument('--backend', choices=['auto', 'osmesa', 'egl'], default='auto')
    snapshot.add_argument('--output', default=str(OUTPUTS_DIR/'insert_plug'/'scene_preview'))
    snapshot.add_argument('--width', type=int, default=960, help='Width and height of each PNG (default: 960)')
    build = sub.add_parser('build', help='Generate the scene XML without running it')
    build.add_argument('--config', default=None)
    build.add_argument('--output', default=str(DEFAULT_SCENE_PATH))
    for name in ('demo', 'collect'):
        p = sub.add_parser(name, help='Execute the physical expert and optionally record dual-camera data')
        p.add_argument('--config', default=None)
        p.add_argument('--output', help='Episode directory (default: outputs/datasets/insert_plug/run_<time>)')
        p.add_argument('--episodes', type=int, default=1 if name == 'demo' else 5)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--randomize-mm', type=float, default=0.)
        p.add_argument('--backend', choices=['auto','osmesa','egl','glfw'], default='auto')
        p.add_argument('--width', type=int, help='Set both camera image dimensions for this run')
        p.add_argument('--control-hz', type=int, help='Override expert target-update rate (default: 10 Hz)')
        p.add_argument('--sample-hz', type=float, help='Override saved observation/action and head video rate (default: 10 Hz)')
        p.add_argument('--no-images', action='store_true', help='Physics-only smoke run; does not create a dataset')
        p.add_argument('--viewer', action='store_true', help='Open interactive viewer; requires a graphical display')
        p.add_argument('--realtime', action='store_true')
        p.add_argument('--quiet', action='store_true')
        p.add_argument('--video', action='store_true',
                       help='Also create a dual-view preview of the first episode; every episode already includes a head MP4')
    d = sub.add_parser('doctor', help='Check physics and rendering backends')
    d.add_argument('--backend', choices=['osmesa','egl','glfw'])
    v = sub.add_parser('validate', help='Check HDF5 schema and observation/action alignment')
    v.add_argument('path')
    r = sub.add_parser('preview', help='Replay recorded cameras to a side-by-side video')
    r.add_argument('path')
    r.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'rollout':
        if args.viewer and sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            parser.error('rollout --viewer requires a graphical desktop (DISPLAY is empty). Use --backend egl for video-only evaluation.')
        return sys.exit(rollout_policy(args))
    if args.command == 'plan':
        return plan_scene(args)
    if args.command == 'snapshot':
        if args.width < 1:
            parser.error('snapshot width must be positive')
        return snapshot_scene(args)
    if args.command == 'view':
        if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            parser.error('view requires a graphical desktop (DISPLAY is empty). Run it in a desktop terminal or configure display forwarding.')
        return view_scene(args)
    if args.command in ('demo','collect'):
        if args.episodes < 1 or args.randomize_mm < 0:
            parser.error('episodes must be positive and randomize-mm must be nonnegative')
        return sys.exit(collect(args))
    if args.command == 'build':
        from .scene import load_config, build_scene
        print(build_scene(load_config(args.config), args.output))
    elif args.command == 'doctor':
        from mujoco_tools.doctor import run_diagnostics
        result = run_diagnostics(chosen_backend=args.backend)
        _print(result)
        sys.exit(0 if result['ok'] else 2)
    elif args.command == 'validate':
        from .dataset import validate_episode
        result = validate_episode(args.path)
        _print(result)
        sys.exit(0 if result['valid'] else 2)
    elif args.command == 'preview':
        from .replay import generate_preview
        print(generate_preview(args.path, args.output))


if __name__ == '__main__':
    main()
