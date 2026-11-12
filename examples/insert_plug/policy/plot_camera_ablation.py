"""Plot camera-ablation results without importing the simulator or model stack.

Usage: python plot_camera_ablation.py --report /path/to/summary.json
Requires NumPy and matplotlib; it can run in a separate plotting environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAMERA_LABELS = {'head': 'Head / third-person', 'wrist': 'Wrist'}
CAMERA_COLORS = {'head': '#2477b7', 'wrist': '#db7c26'}
METRICS = (
    'time_s', 'q0_normalized_rms', 'q0_arm_rms_deg', 'q0_gripper_abs_mm',
    'isolated_ensemble_arm_rms_deg', 'isolated_ensemble_gripper_abs_mm',
    'intermittent_ensemble_arm_rms_deg', 'intermittent_ensemble_gripper_abs_mm',
    'dropout_active',
)


def _series(data, conditions, metric, camera, scheme):
    columns = [item['index'] - 1 for item in conditions
               if item['camera'] == camera and item['scheme'] == scheme]
    if not columns:
        raise ValueError(f'Missing {camera}/{scheme} conditions')
    # Repeated spatial masks first average within each recorded state. Frames
    # are not independent trials; this plot does not attach inferential CIs.
    return data[metric][:, columns].mean(axis=1)


def _load(report_path):
    report_path = Path(report_path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    conditions = report['conditions']
    if [item['index'] for item in conditions] != list(range(len(conditions))):
        raise ValueError('Condition indices must be consecutive, with baseline at index 0')
    if conditions[0]['scheme'] != 'baseline':
        raise ValueError('Condition index 0 must be the unmasked baseline')
    with np.load(report_path.parent / 'data' / 'metrics.npz', allow_pickle=False) as stored:
        data = {key: np.array(stored[key], copy=True) for key in METRICS}
    time_s = data['time_s']
    if time_s.ndim != 1 or not len(time_s) or not np.isfinite(time_s).all():
        raise ValueError('time_s must be a nonempty finite vector')
    if len(time_s) > 1 and not np.all(np.diff(time_s) > 0):
        raise ValueError('Recorded observation times must increase')
    for key in METRICS[1:-1]:
        expected = (len(time_s), 2 if key.startswith('intermittent_') else len(conditions)-1)
        if data[key].shape != expected or not np.isfinite(data[key]).all():
            raise ValueError(f'{key} must be finite with shape {expected}')
        if np.any(data[key] < 0):
            raise ValueError(f'{key} must contain nonnegative action differences')
    dropout = data['dropout_active']
    if dropout.shape != time_s.shape or not np.isin(dropout, [False, True]).all():
        raise ValueError('dropout_active must be one boolean value per observation')
    return report_path, report, data


def plot_report(report_path, output_dir=None):
    """Save a PNG and SVG with descriptive, paired action-sensitivity plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    report_path, report, data = _load(report_path)
    conditions = report['conditions']
    protocol = report.get('protocol', {})
    source = report.get('source', {})
    destination = Path(output_dir).expanduser().resolve() if output_dir else report_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    times = data['time_s']
    patch_fraction = float(protocol.get('patch_fraction', .5))
    dropout_probability = float(protocol.get('drop_probability', .5))
    seed = protocol.get('seed', protocol.get('mask_seed', report.get('seed', 'not recorded')))
    position = source.get('position', 'recorded rollout')
    tick = float(np.median(np.diff(times))) if len(times) > 1 else .1
    end = float(times[-1] + tick)
    patch_label = f'{patch_fraction:.0%} random patches'

    with plt.rc_context({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.titleweight': 'semibold',
        'axes.labelsize': 10, 'axes.edgecolor': '#aeb8c2',
        'axes.labelcolor': '#253445', 'text.color': '#253445',
        'xtick.color': '#46576a', 'ytick.color': '#46576a',
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    }):
        fig, axes = plt.subplots(3, 2, figsize=(15, 12.8))
        for ax in axes.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#d8dee5', linewidth=.7, alpha=.8)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.tick_params(axis='both', labelsize=9)

        ax = axes[0, 0]
        schemes = [('full_mean', 'Full view\nmean fill'),
                   ('patch_mean', f'{patch_fraction:.0%} patches\nmean fill'),
                   ('full_black', 'Full view\nblack fill')]
        available = [(scheme, label) for scheme, label in schemes
                     if all(any(item['camera'] == camera and item['scheme'] == scheme
                                for item in conditions) for camera in CAMERA_LABELS)]
        if not available:
            raise ValueError('No paired head/wrist ablations found')
        x = np.arange(len(available))
        maximum = 0.
        for index, camera in enumerate(CAMERA_LABELS):
            values = [float(_series(data, conditions, 'q0_normalized_rms', camera, scheme).mean())
                      for scheme, _ in available]
            maximum = max(maximum, max(values))
            bars = ax.bar(x + (index-.5)*.34, values, width=.32,
                          color=CAMERA_COLORS[camera], zorder=3)
            for bar, value in zip(bars, values):
                ax.annotate(f'{value:.3g}', (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0, 5), textcoords='offset points', ha='center',
                            va='bottom', fontsize=9, color=CAMERA_COLORS[camera])
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in available])
        ax.set_ylim(0, maximum*1.24 if maximum else 1)
        ax.set_ylabel('RMS change in normalized action units')
        ax.set_title('A  Current prediction: mean sensitivity', loc='left', pad=12)

        def draw_pair(ax, metric, scheme, title, ylabel):
            for camera in CAMERA_LABELS:
                values = _series(data, conditions, metric, camera, scheme)
                ax.plot(times, values, color=CAMERA_COLORS[camera], linewidth=1.6)
            ax.set_title(title, loc='left', pad=12)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Recorded simulation time (s)')
            ax.set_xlim(float(times[0]), end)
            ax.set_ylim(bottom=0)

        draw_pair(axes[0, 1], 'q0_normalized_rms', 'patch_mean',
                  f'B  Current prediction: {patch_label}',
                  'RMS change in normalized action units')
        draw_pair(axes[1, 0], 'q0_arm_rms_deg', 'full_mean',
                  'C  Current prediction: full-view mask', 'Arm target change (RMS degrees)')
        draw_pair(axes[1, 1], 'q0_gripper_abs_mm', 'full_mean',
                  'D  Current prediction: full-view mask', 'Gripper target change (absolute mm)')

        for ax, suffix, title, ylabel in (
            (axes[2, 0], 'arm_rms_deg', 'E  Ensembled action: arm targets', 'Arm target change (RMS degrees)'),
            (axes[2, 1], 'gripper_abs_mm', 'F  Ensembled action: gripper target', 'Gripper target change (absolute mm)'),
        ):
            for index, camera in enumerate(CAMERA_LABELS):
                isolated = _series(data, conditions, f'isolated_ensemble_{suffix}', camera, 'full_mean')
                intermittent = data[f'intermittent_ensemble_{suffix}'][:, index]
                ax.plot(times, isolated, color=CAMERA_COLORS[camera], linewidth=1.1,
                        linestyle='--', alpha=.75)
                ax.plot(times, intermittent, color=CAMERA_COLORS[camera], linewidth=1.6)
            active_times = times[data['dropout_active'].astype(bool)]
            ax.plot(active_times, np.full_like(active_times, .015), linestyle='none', marker='|',
                    markersize=5, color='#a0a9b2', transform=ax.get_xaxis_transform(), alpha=.6)
            ax.set_title(title, loc='left', pad=12)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Recorded simulation time (s)')
            ax.set_xlim(float(times[0]), end)
            ax.set_ylim(bottom=0)

        fig.suptitle(f'ACT camera sensitivity | {position} | {len(times)} recorded states',
                     fontsize=19, fontweight='semibold', y=.984)
        fig.text(.5, .951,
                 f'Paired masks, seed {seed} | {patch_label} | '
                 f'intermittent full-view dropout probability {dropout_probability:.0%}',
                 ha='center', fontsize=10, color='#536477')
        camera_handles = [Line2D([0], [0], color=CAMERA_COLORS[key], linewidth=2.5,
                                 label=label) for key, label in CAMERA_LABELS.items()]
        fig.legend(handles=camera_handles, loc='upper center', bbox_to_anchor=(.5, .938),
                   ncol=2, frameon=False, fontsize=11)
        fig.tight_layout(rect=(.015, .13, .985, .905), h_pad=2.8, w_pad=2.6)
        fig.text(.06, .094,
                 'A–D: change in the newest prediction (query 0). E–F: change after temporal ensembling.',
                 fontsize=10)
        fig.text(.06, .074,
                 'E–F solid: accumulated random-dropout history; dashed: one isolated mask with identical prior history.',
                 fontsize=10)
        fig.text(.06, .054,
                 'Gray ticks: dropout-active steps. All panels use the same recorded states; patch repeats are averaged within each state.',
                 fontsize=9, color='#536477')
        fig.text(.06, .029,
                 'Predicted action-target differences, not measured robot motion or masked closed-loop task success.',
                 fontsize=10, fontweight='semibold')
        paths = {'png': destination / 'summary.png', 'svg': destination / 'summary.svg'}
        try:
            fig.savefig(paths['png'], dpi=170)
            fig.savefig(paths['svg'])
        finally:
            plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, help='Camera-ablation summary.json')
    parser.add_argument('--output-dir', help='Defaults to the directory containing summary.json')
    args = parser.parse_args(argv)
    paths = plot_report(args.report, args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == '__main__':
    main()
