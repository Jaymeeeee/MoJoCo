"""Plot the verified-baseline source-assembly translation diagnostic.

Only NumPy and matplotlib are required; no simulator or policy is imported.
Run ``python -m examples.insert_plug.policy.plot_grasp_translation --output RUN``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODELS = {'act': 'ACT RGB-D', 'diffusion': 'DP RGB-D'}
DIRECTIONS = ('x_plus', 'x_minus', 'y_plus', 'y_minus')
LABELS = {'x_plus': '+X', 'x_minus': '\N{MINUS SIGN}X',
          'y_plus': '+Y', 'y_minus': '\N{MINUS SIGN}Y'}
VECTORS = {'x_plus': np.array([1., 0., 0.]), 'x_minus': np.array([-1., 0., 0.]),
           'y_plus': np.array([0., 1., 0.]), 'y_minus': np.array([0., -1., 0.])}
FK_COLOR = '#bc6b22'
ACTUAL_COLOR = '#246eae'


def _load(output):
    episodes = {}
    for model in MODELS:
        conditions = {}
        for direction in ('baseline', *DIRECTIONS):
            directory = output / 'episodes' / model / direction
            if not (directory / 'result.json').exists():
                continue
            result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            with np.load(directory / 'trace.npz', allow_pickle=False) as data:
                trace = {key: np.array(data[key]) for key in data.files}
            times = trace['after_time']
            if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
                raise ValueError(f'Invalid action timestamps: {directory}')
            if not np.all(np.diff(times) > 0):
                raise ValueError(f'Non-increasing action timestamps: {directory}')
            for key in ('target_tcp', 'after_tcp'):
                if trace[key].shape != (len(times), 3) or not np.isfinite(trace[key]).all():
                    raise ValueError(f'Invalid {key}: {directory}')
            conditions[direction] = {'trace': trace, 'result': result}
        if conditions:
            if 'baseline' not in conditions:
                raise ValueError(f'{model} has shifted episodes but no baseline')
            episodes[model] = conditions
    if not episodes:
        raise ValueError('No completed baseline episodes found')
    return episodes


def _style(axes):
    for ax in np.asarray(axes).flat:
        for name in ('top', 'right'):
            ax.spines[name].set_visible(False)
        ax.grid(axis='y', color='#d9dfe6', linewidth=.6)
        ax.set_axisbelow(True)


def _save(fig, destination, stem, plt):
    paths = []
    try:
        for extension in ('png', 'pdf'):
            path = destination / f'{stem}.{extension}'
            fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
            paths.append(str(path.relative_to(destination.parent)))
    finally:
        plt.close(fig)
    return paths


def _qualification(episode):
    return episode['result']['qualification']


def _event(episode, name='grasp_event'):
    return _qualification(episode).get(name)


def _source_bounds(episode):
    within = episode['result']['within_training_source_bounds']
    return 'within source bounds' if within else 'outside source bounds'


def _baseline_qualification(episodes, destination, plt, Line2D, Patch):
    fig, axes = plt.subplots(1, len(episodes), figsize=(6.3 * len(episodes), 4.9),
                             sharey=True, squeeze=False)
    _style(axes)
    for ax, (model, conditions) in zip(axes[0], episodes.items()):
        baseline = conditions['baseline']
        trace = baseline['trace']
        t = trace['physics_time']
        tip_clearance = (trace['physics_peg_tip'][:, 2] - float(trace['source_rim_z'])) * 1000
        contacts = trace['physics_dual_contact'].astype(bool)
        if tip_clearance.shape != t.shape or contacts.shape != t.shape:
            raise ValueError('Physics timestamps, peg tips and contact arrays must align')
        ax.fill_between(t, 0, 1, where=contacts, step='post',
                        transform=ax.get_xaxis_transform(), color='#79ad86', alpha=.16,
                        linewidth=0)
        ax.plot(t, tip_clearance, color=ACTUAL_COLOR, linewidth=1.8)
        ax.axhline(0, color='#8e9bab', linewidth=.8)
        ax.axhline(5, color='#3e4855', linestyle='--', linewidth=1)
        annotations = []
        for name, label, color in (('grasp_event', 'Grasp', '#9b5bb0'),
                                    ('lift_event', 'Lift', '#3a8c68')):
            event = _event(baseline, name)
            if event:
                start = float(event['start_time_s'])
                confirmed = float(event['confirmed_time_s'])
                ax.axvspan(start, confirmed, color=color, alpha=.12)
                ax.axvline(confirmed, color=color, linewidth=1.2, linestyle=':')
                annotations.append(f'{label} verified: {confirmed:.3f} s')
        status = 'PASS' if _qualification(baseline)['passed'] else 'NOT QUALIFIED'
        ax.set_title(f'{MODELS[model]} | baseline lift {status}', loc='left', fontsize=11)
        ax.text(.02, .97, '\n'.join(annotations), transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8})
        ax.set_xlabel('Elapsed physical time (s)')
        ax.set_xlim(float(t[0]), float(t[-1]))
    axes[0, 0].set_ylabel('Actual screw-tip height above source rim (mm)')
    handles = [Line2D([0], [0], color=ACTUAL_COLOR, label='Measured screw tip'),
               Line2D([0], [0], color='#3e4855', linestyle='--', label='Lift clearance: 5 mm'),
               Patch(facecolor='#79ad86', alpha=.3, label='Opposed finger contact')]
    fig.suptitle('Baseline qualification: actual grasp and extraction', fontsize=15,
                 fontweight='semibold', y=.99)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .84))
    fig.text(.025, .075, 'Qualification uses 1 kHz physics: opposed contact and relative peg/TCP stability '
             'for 200 ms; lift also requires 5 mm rim clearance.', fontsize=8.5)
    fig.text(.025, .035, 'Shaded event windows end at verification. Contact alone and a closed gripper '
             'do not establish successful pickup.', fontsize=8.5)
    return _save(fig, destination, 'baseline_grasp_qualification', plt)


def _paired_arrays(baseline, shifted, direction):
    a, b = baseline['trace'], shifted['trace']
    if a['after_time'].shape != b['after_time'].shape:
        raise ValueError('Paired episodes must cover the same fixed-duration horizon')
    np.testing.assert_allclose(a['after_time'], b['after_time'], rtol=0, atol=1e-9)
    unit = VECTORS[direction]
    return {key: (b[key] - a[key]) @ unit * 1000 for key in ('target_tcp', 'after_tcp')}


def _translation_curves(episodes, destination, plt, Line2D):
    fig, axes = plt.subplots(4, len(episodes), figsize=(6.3 * len(episodes), 10.8),
                             sharex=True, sharey=True, squeeze=False)
    _style(axes)
    for column, (model, conditions) in enumerate(episodes.items()):
        baseline = conditions['baseline']
        event = _event(baseline)
        for row, direction in enumerate(DIRECTIONS):
            ax = axes[row, column]
            ax.axhline(30, color='#3e4855', linestyle='--', linewidth=1)
            ax.axhline(0, color='#8e9bab', linewidth=.7)
            ax.set_title(f'{MODELS[model]} | source shift {LABELS[direction]} 30 mm',
                         loc='left', fontsize=10)
            if direction not in conditions:
                ax.text(.5, .5, 'Not run', transform=ax.transAxes, ha='center')
                continue
            response = _paired_arrays(baseline, conditions[direction], direction)
            t = baseline['trace']['after_time']
            ax.plot(t, response['after_tcp'], color=ACTUAL_COLOR, linewidth=1.6)
            ax.plot(t, response['target_tcp'], color=FK_COLOR, linewidth=1.3, linestyle='--')
            if event:
                ax.axvline(event['confirmed_time_s'], color='#9b5bb0', linestyle=':', linewidth=1.2)
            for time_s in baseline['trace']['chunk_time'][1:]:
                ax.axvline(time_s, color='#cbd2dc', linewidth=.6, zorder=0)
            shifted = conditions[direction]
            status = 'verified' if _qualification(shifted)['passed'] else 'not verified'
            duration = float(shifted['trace']['after_time'][-1])
            ax.text(.015, .035,
                    f'Shifted pickup: {status} within {duration:g} s | {_source_bounds(shifted)}',
                    transform=ax.transAxes, fontsize=8,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85})
            if column == 0:
                ax.set_ylabel('TCP response along\nsource shift (mm)')
            if row == 3:
                ax.set_xlabel('Elapsed physical time (s)')
    handles = [Line2D([0], [0], color=ACTUAL_COLOR, label='Actual TCP'),
               Line2D([0], [0], color=FK_COLOR, linestyle='--', label='FK of commanded joints'),
               Line2D([0], [0], color='#3e4855', linestyle='--', label='Full source shift: 30 mm'),
               Line2D([0], [0], color='#9b5bb0', linestyle=':', label='Baseline verified grasp')]
    fig.suptitle('Closed-loop response to moving the screw and source holder together',
                 fontsize=15, fontweight='semibold', y=.995)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .97),
               ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, .095, 1, .94), h_pad=1.4)
    fig.text(.025, .064, 'Each curve is one paired trial: shifted TCP minus baseline TCP, projected onto '
             'the source-shift direction. No averaging or confidence bands.', fontsize=8.5)
    fig.text(.025, .043, 'Joint targets are absolute and mapped by FK. Later closed-loop observations '
             'differ; 30 mm is a grasp-target reference, not an expectation for the whole path.', fontsize=8.5)
    fig.text(.025, .022, 'Source bounds refer to the training source-XY bounding box: +X/+Y are outside; '
             '\N{MINUS SIGN}X/\N{MINUS SIGN}Y are within. Within bounds does not imply strict interpolation.', fontsize=8.5)
    return _save(fig, destination, 'translation_vs_time', plt)


def _grasp_bars(episodes, destination, plt, Line2D):
    fig, axes = plt.subplots(1, len(episodes), figsize=(6.3 * len(episodes), 5.4),
                             sharey=True, squeeze=False)
    _style(axes)
    for ax, (model, conditions) in zip(axes[0], episodes.items()):
        baseline = conditions['baseline']
        event = _event(baseline)
        ax.axhline(30, color='#3e4855', linestyle='--', linewidth=1)
        ax.axhline(0, color='#8e9bab', linewidth=.7)
        if not event:
            ax.set_title(f'{MODELS[model]} | no verified baseline grasp', loc='left')
            continue
        index = int(event['confirmed_action_index'])
        sampled_time = float(baseline['trace']['after_time'][index])
        event_time = float(event['confirmed_time_s'])
        delay = sampled_time - event_time
        if not -1e-8 <= delay <= .10000001:
            raise ValueError('Grasp event action index must identify its enclosing control interval')
        x, predicted, actual, labels = [], [], [], []
        for direction_index, direction in enumerate(DIRECTIONS):
            if direction not in conditions:
                labels.append(f'{LABELS[direction]}\nnot run')
                continue
            response = _paired_arrays(baseline, conditions[direction], direction)
            x.append(direction_index)
            predicted.append(response['target_tcp'][index])
            actual.append(response['after_tcp'][index])
            status = 'lift OK' if _qualification(conditions[direction])['passed'] else 'no verified lift'
            labels.append(f'{LABELS[direction]}\n{status}\n{_source_bounds(conditions[direction])}')
        x = np.asarray(x, dtype=float)
        bars1 = ax.bar(x - .18, predicted, width=.34, color=FK_COLOR, label='FK target')
        bars2 = ax.bar(x + .18, actual, width=.34, color=ACTUAL_COLOR, label='Actual TCP')
        for bar in [*bars1, *bars2]:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        (bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -3), textcoords='offset points',
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        ax.margins(y=.22)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f'{MODELS[model]} | paired action {index} at {sampled_time:.1f} s\n'
                     f'Baseline grasp verified at {event_time:.3f} s', loc='left', fontsize=11)
    axes[0, 0].set_ylabel('TCP response along source shift (mm)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#3e4855', linestyle='--'))
    labels.append('Full source shift: 30 mm')
    fig.suptitle('Grasp-phase translation response from a physically qualified baseline',
                 fontsize=15, fontweight='semibold', y=.99)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .93),
               ncol=3, frameon=False, fontsize=9)
    duration = max(float(c['baseline']['trace']['after_time'][-1]) for c in episodes.values())
    fig.tight_layout(rect=(0, .21, 1, .84))
    fig.text(.025, .12, 'The baseline verified-grasp action is selected once per model and applied '
             'to every shift. Bars show single trials, not success rates.', fontsize=8.5)
    fig.text(.025, .08, 'Actual TCP is sampled at the end of that 100 ms action, within 100 ms of '
             f'the 1 kHz grasp event. Pickup labels refer to the {duration:g} s trial.', fontsize=8.5)
    fig.text(.025, .04, 'Source bounds are the training source-XY bounding box. '
             'Within bounds does not imply strict interpolation; no verified lift means none within the trial.', fontsize=8.5)
    return _save(fig, destination, 'grasp_translation_response', plt)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output = Path(output).expanduser().resolve()
    episodes = _load(output)
    destination = output / 'plots'
    destination.mkdir(exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.edgecolor': '#a6b1bd', 'text.color': '#27384b',
                         'axes.labelcolor': '#27384b', 'pdf.fonttype': 42}):
        paths = _baseline_qualification(episodes, destination, plt, Line2D, Patch)
        if any(any(d in conditions for d in DIRECTIONS) for conditions in episodes.values()):
            paths += _translation_curves(episodes, destination, plt, Line2D)
            paths += _grasp_bars(episodes, destination, plt, Line2D)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps({'plots': plot_results(args.output)}, indent=2))


if __name__ == '__main__':
    main()
