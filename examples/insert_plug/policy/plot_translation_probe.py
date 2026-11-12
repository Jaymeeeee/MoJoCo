"""Plot paired translation probes without importing MuJoCo or the policy stack.

Run ``python -m examples.insert_plug.policy.plot_translation_probe --output RUN``.
The plots describe predicted TCP targets, not executed motion or grasp success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODEL_LABELS = {'act': 'ACT RGB-D', 'diffusion': 'Diffusion RGB-D'}
CONDITION_LABELS = {'peg_and_holder': 'Object + source holder'}
MODEL_COLORS = {'act': '#246eae', 'diffusion': '#cb702a'}
DIRECTIONS = ('x_plus', 'x_minus', 'y_plus', 'y_minus')
DIRECTION_LABELS = {'x_plus': '+X', 'x_minus': '\N{MINUS SIGN}X',
                    'y_plus': '+Y', 'y_minus': '\N{MINUS SIGN}Y'}
DIRECTION_MARKERS = {'x_plus': '>', 'x_minus': '<', 'y_plus': '^', 'y_minus': 'v'}
DIRECTION_COLORS = {'x_plus': '#256da9', 'x_minus': '#df7b27',
                    'y_plus': '#3a8c68', 'y_minus': '#9b5bb0'}


def _load(output: Path):
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    rows = summary['rows']
    if not rows:
        raise ValueError('Translation summary contains no paired rows')
    loaded = []
    for row in rows:
        archive = (output / row['archive']).resolve()
        try:
            archive.relative_to(output)
        except ValueError as exc:
            raise ValueError('Pair archive must be inside the experiment output directory') from exc
        with np.load(archive, allow_pickle=False) as data:
            baseline = np.asarray(data['baseline_tcp'], dtype=float)
            shifted = np.asarray(data['shifted_tcp'], dtype=float)
            delta = np.asarray(data['delta'], dtype=float)
        if (baseline.ndim != 2 or baseline.shape[1] != 3 or not len(baseline)
                or shifted.shape != baseline.shape or delta.shape != (3,)
                or not all(np.isfinite(a).all() for a in (baseline, shifted, delta))):
            raise ValueError(f'Invalid paired TCP trajectories: {archive}')
        distance = float(np.linalg.norm(delta))
        if distance <= 0:
            raise ValueError(f'Object translation must be nonzero: {archive}')
        for field, allowed in (('model', MODEL_LABELS), ('condition', CONDITION_LABELS),
                               ('direction', DIRECTIONS)):
            if row[field] not in allowed:
                raise ValueError(f'Unknown {field}: {row[field]!r}')
        loaded.append(dict(row=row, along_mm=(shifted-baseline) @ (delta/distance)*1000,
                           distance_mm=distance*1000))
    return summary, loaded


def _save(fig, destination: Path, stem: str, plt) -> list[str]:
    paths = []
    try:
        for suffix in ('png', 'pdf'):
            path = destination / f'{stem}.{suffix}'
            fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
            paths.append(str(path.relative_to(destination.parent)))
    finally:
        plt.close(fig)
    return paths


def _style_axes(axes):
    for ax in np.asarray(axes).flat:
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.grid(axis='y', color='#d9dfe6', linewidth=.6)
        ax.set_axisbelow(True)


def _curves(loaded, models, destination, plt, Line2D):
    fig, axes = plt.subplots(len(DIRECTIONS), len(models),
                             figsize=(6.5*len(models), 10.5), squeeze=False,
                             sharex=True, sharey=True)
    _style_axes(axes)
    for column, model in enumerate(models):
        for row_index, direction in enumerate(DIRECTIONS):
            ax = axes[row_index, column]
            selected = [p for p in loaded if p['row']['model'] == model
                        and p['row']['direction'] == direction]
            targets = sorted({round(p['distance_mm'], 6) for p in selected})
            for target in targets:
                ax.axhline(target, color='#36404b', linestyle='--', linewidth=1)
            ax.axhline(0, color='#8995a4', linewidth=.7)
            counts = []
            for condition in CONDITION_LABELS:
                color = MODEL_COLORS[model]
                pairs = [p for p in selected if p['row']['condition'] == condition]
                if not pairs:
                    continue
                shapes = {p['along_mm'].shape for p in pairs}
                if len(shapes) != 1:
                    raise ValueError('Cannot average trajectories with different chunk lengths')
                values = np.stack([p['along_mm'] for p in pairs])
                query = np.arange(values.shape[1])
                ax.fill_between(query, values.min(axis=0), values.max(axis=0),
                                color=color, alpha=.13, linewidth=0)
                ax.plot(query, values.mean(axis=0), color=color, linewidth=1.7)
                counts.append(f'{len(pairs)} paired predictions')
            ax.set_title(f'{MODEL_LABELS[model]} | translation {DIRECTION_LABELS[direction]}',
                         loc='left', fontsize=11)
            if counts:
                ax.text(.015, .035, '\n'.join(counts), transform=ax.transAxes,
                        fontsize=7.5, color='#536273')
            else:
                ax.text(.5, .5, 'No paired observations', transform=ax.transAxes,
                        ha='center', va='center')
            if column == 0:
                ax.set_ylabel('TCP target displacement\nalong object shift (mm)')
            if row_index == len(DIRECTIONS)-1:
                ax.set_xlabel('Predicted action query (zero-based)')
    target_values = sorted({round(p['distance_mm'], 6) for p in loaded})
    reference_label = (f'Object shift: {target_values[0]:g} mm' if len(target_values) == 1
                       else 'Object shift magnitude')
    handles = [Line2D([0], [0], color=MODEL_COLORS[key], linewidth=2, label=MODEL_LABELS[key])
               for key in models]
    handles.append(Line2D([0], [0], color='#36404b', linestyle='--', label=reference_label))
    fig.suptitle('Predicted TCP response to joint translation of object and source holder',
                 fontsize=16, fontweight='semibold', y=.998)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .971),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .067, 1, .94), h_pad=1.3)
    fig.text(.035, .038,
             'Lines: mean over paired layouts / sampling seeds. Bands: observed min-max, not confidence intervals.',
             fontsize=9, color='#455566')
    fig.text(.035, .017,
             'Absolute joint targets are mapped by forward kinematics. This is not measured robot motion or grasp success.',
             fontsize=9, color='#455566')
    return _save(fig, destination, 'translation_vs_horizon', plt)


def _endpoint_scatter(loaded, models, destination, plt, Line2D):
    endpoints = (('last', 'Last action query'),
                 ('baseline_first_closure', 'Baseline closure query (commanded aperture proxy)'))
    panels = [(name, title, condition) for name, title in endpoints
              for condition in CONDITION_LABELS]
    fig, axes = plt.subplots(len(panels), len(models),
                             figsize=(6.5*len(models), 6.8), squeeze=False,
                             sharex=True, sharey=True)
    _style_axes(axes)
    positions = list(dict.fromkeys(p['row']['position'] for p in loaded))
    offsets = dict(zip(DIRECTIONS, (-.24, -.08, .08, .24)))
    any_values = False
    for row_index, (endpoint, title, condition) in enumerate(panels):
        for column, model in enumerate(models):
            ax = axes[row_index, column]
            ax.axhline(1, color='#36404b', linewidth=1, linestyle='--')
            ax.axhline(0, color='#8995a4', linewidth=.7)
            available = missing = 0
            for direction in DIRECTIONS:
                for position_index, position in enumerate(positions):
                    pairs = [p for p in loaded if p['row']['model'] == model
                             and p['row']['condition'] == condition
                             and p['row']['direction'] == direction
                             and p['row']['position'] == position]
                    values = []
                    for pair in pairs:
                        value = pair['row'].get('metrics', {}).get('endpoints', {}).get(endpoint)
                        if value is None:
                            missing += 1
                            continue
                        gain = float(value['gain'])
                        if not np.isfinite(gain):
                            raise ValueError('Endpoint gain must be finite or endpoint must be null')
                        values.append(gain)
                    if not values:
                        continue
                    any_values = True
                    available += len(values)
                    x = position_index+offsets[direction]
                    jitter = np.linspace(-.025, .025, len(values)) if len(values) > 1 else [0.]
                    color = DIRECTION_COLORS[direction]
                    ax.scatter(x+np.asarray(jitter), values, s=15, color=color, alpha=.28,
                               marker=DIRECTION_MARKERS[direction], linewidths=0)
                    ax.plot([x, x], [min(values), max(values)], color=color, linewidth=1, alpha=.6)
                    ax.scatter([x], [np.mean(values)], s=38, color=color,
                               marker=DIRECTION_MARKERS[direction], edgecolors='white', linewidths=.4)
            ax.set_title(f'{MODEL_LABELS[model]} | {CONDITION_LABELS[condition]}\n{title}',
                         loc='left', fontsize=10)
            ax.text(.015, .04, f'{available} paired endpoints; {missing} unavailable',
                    transform=ax.transAxes, fontsize=8, color='#536273')
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels([str(p) for p in positions], rotation=35, ha='right')
            ax.set_xlim(-.6, len(positions)-.4)
            if column == 0:
                ax.set_ylabel('Signed translation gain\n(1 = full object shift)')
            if row_index == len(panels)-1:
                ax.set_xlabel('Initial layout')
    handles = [Line2D([0], [0], color=DIRECTION_COLORS[d], marker=DIRECTION_MARKERS[d],
                      linestyle='none', label=DIRECTION_LABELS[d]) for d in DIRECTIONS]
    handles.append(Line2D([0], [0], color='#36404b', linestyle='--', label='Gain = 1'))
    fig.suptitle('Directional response of predicted TCP targets',
                 fontsize=16, fontweight='semibold', y=.998)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .972),
               ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, .93), h_pad=1.5)
    fig.text(.035, .043,
             'Small points: individual paired seeds. Large points: mean. Vertical segments: observed min-max.', fontsize=9)
    fig.text(.035, .023,
             'Closure uses the same baseline-selected query for both conditions; it does not establish physical grasp.', fontsize=9)
    if not any_values:
        plt.close(fig)
        return []
    return _save(fig, destination, 'endpoint_translation_gain', plt)


def plot_results(output: Path) -> list[str]:
    """Save descriptive PNG/PDF figures under RUN/plots; return relative paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output).expanduser().resolve()
    _, loaded = _load(output)
    models = [model for model in MODEL_LABELS if any(p['row']['model'] == model for p in loaded)]
    destination = output / 'plots'
    destination.mkdir(exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.edgecolor': '#a6b1bd', 'text.color': '#27384b',
                         'axes.labelcolor': '#27384b', 'pdf.fonttype': 42}):
        return (_curves(loaded, models, destination, plt, Line2D)
                + _endpoint_scatter(loaded, models, destination, plt, Line2D)
                + _execution_comparison(loaded, models, destination, plt, Line2D))


def _execution_comparison(loaded, models, destination, plt, Line2D):
    """Keep model-requested translation separate from realized servo motion."""
    if not any(item['row'].get('realized_metrics') for item in loaded):
        return []
    endpoints = [('baseline_first_closure', 'Baseline closure command'),
                 ('last', 'After the complete 64-action chunk')]
    fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 10), squeeze=False)
    _style_axes(axes)
    for row_index, (endpoint, label) in enumerate(endpoints):
        for column, model in enumerate(models):
            ax = axes[row_index, column]
            all_values = [0., 30.]
            counts = 0
            for direction in DIRECTIONS:
                values = []
                for item in loaded:
                    row = item['row']
                    if row['model'] != model or row['direction'] != direction:
                        continue
                    predicted = row['metrics']['endpoints'][endpoint]
                    measured = (row.get('realized_metrics') or {}).get('endpoints', {}).get(endpoint)
                    if predicted is not None and measured is not None:
                        values.append([predicted['along_m'] * 1000, measured['along_m'] * 1000])
                if values:
                    points = np.asarray(values)
                    ax.scatter(points[:, 0], points[:, 1], color=DIRECTION_COLORS[direction],
                               marker=DIRECTION_MARKERS[direction], s=24, alpha=.55, linewidths=0)
                    all_values.extend(points.ravel().tolist())
                    counts += len(points)
            low, high = min(all_values) - 3., max(all_values) + 3.
            ax.plot([low, high], [low, high], color='#8995a4', linewidth=.8)
            ax.axvline(30, color='#36404b', linewidth=.8, linestyle='--')
            ax.axhline(30, color='#36404b', linewidth=.8, linestyle='--')
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'{MODEL_LABELS[model]} | {label}', loc='left', fontsize=10)
            ax.text(.025, .94, f'{counts} paired endpoints', transform=ax.transAxes, fontsize=8)
            ax.set_xlabel('Predicted FK displacement along shift (mm)')
            ax.set_ylabel('Executed TCP displacement along shift (mm)')
    handles = [Line2D([0], [0], color=DIRECTION_COLORS[d], marker=DIRECTION_MARKERS[d],
                      linestyle='none', label=DIRECTION_LABELS[d]) for d in DIRECTIONS]
    fig.suptitle('Predicted targets and physical execution respond differently', fontsize=15, y=.99)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .966), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .065, 1, .94), h_pad=2)
    fig.text(.035, .035, 'Diagonal: executed displacement equals the model target displacement. Dashed lines: 30 mm object shift.', fontsize=9)
    fig.text(.035, .015, 'Physical TCP is sampled after each 100 ms command interval. Closure is a commanded-aperture proxy, not grasp success.', fontsize=8)
    return _save(fig, destination, 'predicted_vs_executed', plt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Experiment directory containing summary.json')
    args = parser.parse_args(argv)
    print(json.dumps(plot_results(args.output), indent=2))


if __name__ == '__main__':
    main()
