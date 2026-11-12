"""Plot absolute initial-chunk TCP predictions and paired translation responses.

Requires only NumPy and matplotlib. Statistics are computed across matched
sampling seeds at each query, never across different axes, queries or shifts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODELS = {'act': 'ACT RGB-D', 'diffusion': 'DP RGB-D'}
CONDITIONS = ('baseline', 'x_plus', 'x_minus', 'y_plus', 'y_minus')
LABELS = {'baseline': 'Original', 'x_plus': '+X 30 mm', 'x_minus': '\N{MINUS SIGN}X 30 mm',
          'y_plus': '+Y 30 mm', 'y_minus': '\N{MINUS SIGN}Y 30 mm'}
COLORS = {'baseline': '#55616f', 'x_plus': '#2472b6', 'x_minus': '#d47a23',
          'y_plus': '#2d9466', 'y_minus': '#965eb5'}
MODEL_COLORS = {'act': '#246eae', 'diffusion': '#cb702a'}
QUERIES = np.arange(64)


def load_data(output):
    """Reject incomplete, mismatched or nonfinite input rather than plotting it."""
    models = {}
    for model in MODELS:
        conditions = {}
        for condition in CONDITIONS:
            path = output/'data'/model/f'{condition}.npz'
            with np.load(path, allow_pickle=False) as archive:
                row = {key: np.asarray(archive[key]) for key in
                       ('tcp', 'actions', 'seeds', 'delta', 'source_xyz')}
            n = len(row['seeds'])
            if (n < 1 or row['tcp'].shape != (n, 64, 3)
                    or row['actions'].shape != (n, 64, 8)
                    or row['delta'].shape != (3,) or row['source_xyz'].shape != (3,)
                    or any(not np.isfinite(value).all() for value in row.values())):
                raise ValueError(f'Invalid initial64 array shapes/values: {path}')
            if len(np.unique(row['seeds'])) != n:
                raise ValueError(f'Duplicate sampling seeds: {path}')
            if condition == 'baseline':
                if not np.array_equal(row['delta'], np.zeros(3)):
                    raise ValueError('Baseline translation must be zero')
            else:
                expected = np.zeros(3)
                expected[0 if condition.startswith('x') else 1] = (
                    .03 if condition.endswith('plus') else -.03)
                np.testing.assert_allclose(row['delta'], expected, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(row['seeds'], conditions['baseline']['seeds'])
                np.testing.assert_allclose(row['source_xyz']-conditions['baseline']['source_xyz'],
                                           expected, rtol=0, atol=1e-9)
            conditions[condition] = row
        models[model] = conditions
    return models


def stats(values):
    return dict(mean=np.mean(values, axis=0), minimum=np.min(values, axis=0),
                maximum=np.max(values, axis=0))


def _style(axes):
    for ax in np.asarray(axes).flat:
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#d9dfe6', linewidth=.6)
        ax.set_xlim(0, 63)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 63])
        ax.axvline(31.5, color='#b8c1ca', linewidth=.9, linestyle=':')


def _curve(ax, values, color, label=None, baseline=False):
    summary = stats(values)
    if len(values) > 1:
        ax.fill_between(QUERIES, summary['minimum'], summary['maximum'],
                        color=color, alpha=.13 if baseline else .18, linewidth=0)
    ax.plot(QUERIES, summary['mean'], color=color, label=label,
            linestyle='--' if baseline else '-', linewidth=1.7 if baseline else 2.)
    return summary


def _save(fig, output, stem, plt):
    paths = []
    for suffix in ('png', 'pdf'):
        path = output/'plots'/f'{stem}.{suffix}'
        fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        paths.append(str(path.relative_to(output)))
    plt.close(fig)
    return paths


def _sampling_text(model, n):
    return ('n=1 deterministic prediction; min = mean = max'
            if model == 'act' and n == 1 else f'n={n} sampling seeds; band = observed min\u2013max')


def absolute_xyz(models, output, plt, Line2D):
    fig, axes = plt.subplots(2, 3, figsize=(15.9, 8.3), sharex=True, sharey='col', squeeze=False)
    _style(axes)
    for row, (model, conditions) in enumerate(models.items()):
        n = len(conditions['baseline']['seeds'])
        for column, axis in enumerate('XYZ'):
            ax = axes[row, column]
            for condition, data in conditions.items():
                _curve(ax, data['tcp'][:, :, column]*1000, COLORS[condition],
                       baseline=condition == 'baseline')
            ax.set_title(f'{MODELS[model]} | world {axis}', loc='left', fontsize=11)
            ax.set_ylabel(f'Absolute TCP {axis} (mm)')
            ax.text(.025, .045, 'n=1; min = mean = max' if n == 1 else f'n={n}; mean + min\u2013max',
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            if row == 1:
                ax.set_xlabel('Predicted action query (zero-based)')
    handles = [Line2D([0], [0], color=COLORS[c], linestyle='--' if c == 'baseline' else '-',
                      linewidth=2, label=LABELS[c]) for c in CONDITIONS]
    fig.suptitle('Initial 64-step predictions: absolute TCP coordinates in the world frame',
                 fontsize=16, fontweight='semibold', y=.99)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .945), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .145, 1, .89))
    fig.text(.025, .09, 'Each line is a per-query mean across seeds for one fixed scene. '
             'DP shading is its observed min\u2013max; ACT is deterministic with one prediction.', fontsize=9)
    fig.text(.025, .055, 'Absolute joint commands are mapped through FK. These are initial-observation '
             'forecasts; only seed20 has a physically verified baseline rollout.', fontsize=9)
    fig.text(.025, .02, 'Dotted boundary: n32 deployment executes only q0\u201331 before replanning; '
             'q32\u201363 from this initial prediction were not executed as that chunk.', fontsize=9)
    return _save(fig, output, 'absolute_tcp_xyz_64', plt)


def absolute_shift_axis(models, output, plt, Line2D):
    fig, axes = plt.subplots(4, 2, figsize=(13.3, 11.9), sharex=True, sharey='row', squeeze=False)
    _style(axes)
    for column, (model, conditions) in enumerate(models.items()):
        baseline = conditions['baseline']
        n = len(baseline['seeds'])
        for row, condition in enumerate(CONDITIONS[1:]):
            ax = axes[row, column]
            index = 0 if condition.startswith('x') else 1
            axis = 'XYZ'[index]
            data = conditions[condition]
            _curve(ax, baseline['tcp'][:, :, index]*1000, COLORS['baseline'], baseline=True)
            summary = _curve(ax, data['tcp'][:, :, index]*1000, MODEL_COLORS[model])
            source = float(data['source_xyz'][index])*1000
            ax.axhline(source, color='#bc4e42', linewidth=1.1, linestyle=':')
            ax.set_title(f'{MODELS[model]} | source {LABELS[condition]}', loc='left', fontsize=11)
            ax.text(.02, .97, f'q63: {summary["mean"][-1]:.2f} '
                    f'[{summary["minimum"][-1]:.2f}, {summary["maximum"][-1]:.2f}] mm\n'
                    f'Object world {axis}: {source:.2f} mm',
                    transform=ax.transAxes, va='top', fontsize=8.5,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
            if column == 0:
                ax.set_ylabel(f'Absolute TCP world {axis} (mm)')
            ax.text(.02, .025, 'n=1; min = mean = max' if n == 1 else f'n={n} sampling seeds',
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            if row == 3:
                ax.set_xlabel('Predicted action query (zero-based)')
    handles = [Line2D([0], [0], color=MODEL_COLORS['act'], linewidth=2, label='ACT shifted mean'),
               Line2D([0], [0], color=MODEL_COLORS['diffusion'], linewidth=2, label='DP shifted mean'),
               Line2D([0], [0], color=COLORS['baseline'], linestyle='--', label='Original-position mean'),
               Line2D([0], [0], color='#bc4e42', linestyle=':', label='New object coordinate')]
    fig.suptitle('Absolute 64-step TCP predictions at the four shifted source positions',
                 fontsize=15, fontweight='semibold', y=.995)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .968), ncol=4,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .935))
    fig.text(.025, .067, 'World X is shown for \u00b1X shifts, world Y for \u00b1Y shifts; '
             'negative-axis panels retain world-coordinate signs. Full XYZ are in the companion figure/CSV.', fontsize=8.7)
    fig.text(.025, .042, 'Bands are per-query observed min\u2013max over DP seeds, not confidence intervals. '
             'ACT n=1 has no nonzero uncertainty band. Object lines refer only to the shown axis.', fontsize=8.7)
    fig.text(.025, .017, 'FK forecasts from the same initial robot state, not an executed 64-step rollout. '
             'Only seed20 has a verified baseline rollout; n32 uses q0\u201331 before replanning.', fontsize=8.7)
    return _save(fig, output, 'absolute_shift_axis_64', plt)


def paired_response(models, output, plt, Line2D):
    fig, axes = plt.subplots(4, 2, figsize=(13.3, 10.9), sharex=True, sharey=True, squeeze=False)
    _style(axes)
    for column, (model, conditions) in enumerate(models.items()):
        baseline = conditions['baseline']
        n = len(baseline['seeds'])
        for row, condition in enumerate(CONDITIONS[1:]):
            ax = axes[row, column]
            data = conditions[condition]
            delta = data['delta']
            distance = float(np.linalg.norm(delta))
            values = (data['tcp']-baseline['tcp']) @ (delta/distance)*1000
            _curve(ax, values, MODEL_COLORS[model])
            ax.axhline(distance*1000, color='#3e4855', linestyle='--', linewidth=1.1)
            ax.axhline(0, color='#8795a4', linewidth=.8)
            ax.set_title(f'{MODELS[model]} | source {LABELS[condition]}', loc='left', fontsize=11)
            ax.text(.02, .035, f'One layout; {n} paired prediction' + ('' if n == 1 else 's'),
                    transform=ax.transAxes, fontsize=8.3,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
            if column == 0:
                ax.set_ylabel('Predicted TCP response\nalong source shift (mm)')
            if row == 3:
                ax.set_xlabel('Predicted action query (zero-based)')
    handles = [Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2, label=label)
               for m, label in MODELS.items()]
    handles.append(Line2D([0], [0], color='#3e4855', linestyle='--', label='Source shift: 30 mm'))
    fig.suptitle('Initial 64-step TCP response: shifted prediction minus original prediction',
                 fontsize=15, fontweight='semibold', y=.995)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .965), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, .93))
    fig.text(.025, .065, 'Mean and observed min\u2013max are computed across matched seeds at each query. '
             'ACT n=1; DP n=5. No averaging across layouts, directions or query indices.', fontsize=8.7)
    fig.text(.025, .04, 'Negative values mean the shifted target differs from the original in the opposite '
             'direction to the source shift; this is not the direction of the whole robot trajectory.', fontsize=8.7)
    fig.text(.025, .015, 'Initial-observation FK forecasts, not executed motion. The qualified origin was '
             'verified under n32 closed-loop control; q32\u201363 here are unexecuted initial forecasts.', fontsize=8.7)
    return _save(fig, output, 'translation_vs_horizon_64', plt)


def export_statistics(models, output):
    rows = []
    endpoints = dict(units='millimetres in world frame', endpoint_query=63,
                     pointwise_statistics_axis='sampling seeds at fixed query, scene and XYZ axis',
                     horizon_statistics_axis='all 64 query indices and sampling seeds; descriptive coordinates, not uncertainty',
                     bands='observed minimum/maximum, not confidence intervals', models={})
    for model, conditions in models.items():
        endpoints['models'][model] = {}
        baseline = conditions['baseline']
        for condition, data in conditions.items():
            values = data['tcp']*1000
            summary = stats(values)
            e = dict(seeds=data['seeds'].astype(int).tolist(), n=len(values),
                     source_xyz_mm=(data['source_xyz']*1000).tolist(),
                     endpoint_xyz_mm={key: value[-1].tolist() for key, value in summary.items()},
                     horizon_coordinate_summary_mm=dict(mean=values.mean(axis=(0, 1)).tolist(),
                         minimum=values.min(axis=(0, 1)).tolist(), maximum=values.max(axis=(0, 1)).tolist()))
            for query in QUERIES:
                for index, axis in enumerate('XYZ'):
                    rows.append(dict(model=model, condition=condition, query=int(query), axis=axis,
                        n=len(values), mean_mm=float(summary['mean'][query, index]),
                        minimum_mm=float(summary['minimum'][query, index]),
                        maximum_mm=float(summary['maximum'][query, index]),
                        source_axis_mm=float(data['source_xyz'][index]*1000)))
            if condition != 'baseline':
                delta = data['delta']
                response = (data['tcp']-baseline['tcp']) @ (delta/np.linalg.norm(delta))*1000
                e['paired_q63_response_mm'] = {key: float(value[-1]) for key, value in stats(response).items()}
            endpoints['models'][model][condition] = e
    with (output/'pointwise_absolute_tcp_statistics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'absolute_tcp_statistics.json').write_text(json.dumps(endpoints, indent=2, allow_nan=False)+'\n')
    return endpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    models = load_data(output)
    (output/'plots').mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                         'axes.labelcolor': '#334155', 'text.color': '#334155',
                         'axes.edgecolor': '#a5b1bf'})
    paths = []
    paths += absolute_shift_axis(models, output, plt, Line2D)
    paths += absolute_xyz(models, output, plt, Line2D)
    paths += paired_response(models, output, plt, Line2D)
    export_statistics(models, output)
    print(json.dumps(dict(plots=paths, statistics=['pointwise_absolute_tcp_statistics.csv',
                                                 'absolute_tcp_statistics.json']), indent=2))


if __name__ == '__main__':
    main()
