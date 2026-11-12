"""Describe paired translation-probe results using NumPy and the standard library.

Only read a completed experiment. Write aggregate.json, paired_metrics.csv and
report.md alongside summary.json; do not modify model or inference artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


# Defined before reading experiment results; descriptive tolerance, not a
# significance test, successful-grasp criterion, or tuned model-selection rule.
GAIN_TOLERANCE = .20
ORTHOGONAL_TOLERANCE_M = .005
ENDPOINTS = ('baseline_first_closure', 'independent_first_closures',
             'query31', 'last', 'baseline_minimum_z')
SOURCES = ('metrics', 'realized_metrics')
DIRECTIONS = ('all', 'x_plus', 'x_minus', 'y_plus', 'y_minus')
SCOPES = ('all', 'within_bounds', 'outside_bounds')
FIELDS = {'along_mm': ('along_m', 1000.), 'gain': ('gain', 1.),
          'gain_abs_deviation': ('gain', 1.),
          'endpoint_time_difference_s': ('shifted_command_time_s', 1.),
          'orthogonal_mm': ('orthogonal_m', 1000.), 'residual_mm': ('residual_m', 1000.),
          'baseline_grasp_xy_error_mm': ('baseline_grasp_xy_error_m', 1000.),
          'shifted_grasp_xy_error_mm': ('shifted_grasp_xy_error_m', 1000.)}
MODEL_NAMES = {'act': 'ACT RGB-D', 'diffusion': 'DP RGB-D'}
ENDPOINT_NAMES = {'baseline_first_closure': '基准闭爪时刻（同 query）',
                  'independent_first_closures': '各自闭爪时刻（query 可不同）',
                  'query31': 'query 31（第 32 个动作）', 'last': '完整 chunk 最后动作',
                  'baseline_minimum_z': '基准最低 TCP 高度时刻'}
SCOPE_NAMES = {'all': '全部', 'within_bounds': '源位置逐轴训练范围内',
               'outside_bounds': '源位置逐轴训练范围外'}


def _statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, median=None, min=None, max=None)
    if not np.isfinite(values).all():
        raise ValueError('Cannot summarize nonfinite endpoint metrics')
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                min=float(values.min()), max=float(values.max()))


def _endpoint(row, source, name):
    metric = row.get(source)
    return metric.get('endpoints', {}).get(name) if metric is not None else None


def _selected(row, model, scope, direction):
    return (row['model'] == model
            and (direction == 'all' or row['direction'] == direction)
            and (scope == 'all' or row['within_training_source_bounds'] == (scope == 'within_bounds')))


def _group(rows, source, endpoint):
    grouped = {}
    all_passes = {}
    raw = {field: [] for field in FIELDS}
    raw_passes = []
    missing = 0
    for row in rows:
        value = _endpoint(row, source, endpoint)
        requested = all_passes.setdefault(str(row['position']), {}).setdefault(row['direction'], [])
        if value is None:
            missing += 1
            requested.append(0.)
            continue
        record = {field: float(value[key])*scale for field, (key, scale) in FIELDS.items()}
        record['gain_abs_deviation'] = abs(record['gain']-1.)
        record['endpoint_time_difference_s'] -= float(value['base_command_time_s'])
        passed = (abs(record['gain']-1.) <= GAIN_TOLERANCE + 1e-12
                  and float(value['orthogonal_m']) <= ORTHOGONAL_TOLERANCE_M + 1e-12)
        record['within_descriptive_tolerance'] = float(passed)
        requested.append(float(passed))
        grouped.setdefault(str(row['position']), {}).setdefault(row['direction'], []).append(record)
        for field in FIELDS:
            raw[field].append(record[field])
        raw_passes.append(passed)
    layouts = []
    for position, directions in grouped.items():
        direction_means = []
        for direction, records in directions.items():
            means = {field: float(np.mean([record[field] for record in records]))
                     for field in (*FIELDS, 'within_descriptive_tolerance')}
            direction_means.append(means)
        values = {field: float(np.mean([record[field] for record in direction_means]))
                  for field in (*FIELDS, 'within_descriptive_tolerance')}
        layouts.append(dict(position=position, available_directions=len(directions),
                            available_seed_direction_rows=sum(map(len, directions.values())), **values))
    return dict(
        paired_rows=len(rows), available_rows=len(rows)-missing, missing_rows=missing,
        layouts_total=len({str(row['position']) for row in rows}), layouts_available=len(layouts),
        statistics={field: _statistics([layout[field] for layout in layouts]) for field in FIELDS},
        raw_row_range={field: dict(min=min(values) if values else None, max=max(values) if values else None)
                       for field, values in raw.items()},
        descriptive_tolerance=dict(
            available_only_layout_weighted_fraction=(float(np.mean([v['within_descriptive_tolerance'] for v in layouts]))
                                                     if layouts else None),
            all_requested_layout_weighted_fraction=(float(np.mean([
                np.mean([np.mean(values) for values in directions.values()])
                for directions in all_passes.values()])) if all_passes else None),
            missing_endpoint_rule='Missing endpoints do not satisfy the criterion; all-requested fraction includes them in its denominator. Available-only fraction is separately labeled.',
            raw_rows_satisfying=int(sum(raw_passes)), available_rows=len(raw_passes)),
        layout_means=layouts,
    )


def _closure_counts(rows):
    counts = dict(paired_rows=len(rows), baseline_missing=0, shifted_missing=0,
                  either_missing=0, both_present=0)
    for row in rows:
        closure = row['metrics']['closure']
        base, shifted = bool(closure['baseline_found']), bool(closure['shifted_found'])
        counts['baseline_missing'] += not base
        counts['shifted_missing'] += not shifted
        counts['either_missing'] += not (base and shifted)
        counts['both_present'] += base and shifted
    counts['semantics'] = 'Paired-row counts; a baseline is shared by four directions, not four independent baselines.'
    return counts


def _replays(output, rows, requested, expected_steps):
    unique = {}
    for row in rows:
        directory = (output / row['archive']).parent
        for direction in ('baseline', row['direction']):
            key = (row['model'], str(row['position']), row['seed'], direction)
            unique[key] = directory / f'{direction}.npz'
    result = {}
    for model in dict.fromkeys(row['model'] for row in rows):
        entries = [(key, path) for key, path in unique.items() if key[0] == model]
        counts = dict(requested=bool(requested), unique_predictions=len(entries),
                      completed=0, incomplete=0, unavailable=0, failures=[])
        if not requested:
            result[model] = counts
            continue
        for key, path in entries:
            sidecar = path.with_suffix('.json')
            if not path.is_file() or not sidecar.is_file():
                counts['unavailable'] += 1
                continue
            details = json.loads(sidecar.read_text(encoding='utf-8'))
            with np.load(path, allow_pickle=False) as data:
                size = len(data['realized_tcp']) if 'realized_tcp' in data else None
            failure = details.get('replay_failure')
            if size == expected_steps and failure is None:
                counts['completed'] += 1
            else:
                counts['incomplete'] += 1
                counts['failures'].append(dict(position=key[1], seed=key[2], direction=key[3],
                                               completed_steps=size, failure=failure))
        counts['paired_rows_with_both_complete'] = sum(
            row['model'] == model and row.get('realized_metrics') is not None for row in rows)
        counts['completion_is_not_grasp_success'] = True
        result[model] = counts
    return result


def _csv(output, rows):
    records = []
    for row in rows:
        for source in SOURCES:
            for endpoint in ENDPOINTS:
                value = _endpoint(row, source, endpoint)
                record = {key: row[key] for key in
                          ('model', 'position', 'seed', 'condition', 'direction', 'within_training_source_bounds', 'archive')}
                record.update(metric_source=source, endpoint=endpoint, available=value is not None)
                for field in ('source_xy', 'target_xy'):
                    xy = row.get(field, [None, None])
                    record[field.replace('_xy', '_x_m')] = xy[0]
                    record[field.replace('_xy', '_y_m')] = xy[1]
                if value is not None:
                    for key, item in value.items():
                        if isinstance(item, (str, bool, int, float)) or item is None:
                            record[key] = item
                        elif isinstance(item, list):
                            for index, element in enumerate(item):
                                record[f'{key}_{index}'] = element
                    for field, (key, scale) in FIELDS.items():
                        record[field] = float(value[key])*scale
                    record['gain_abs_deviation'] = abs(float(value['gain'])-1.)
                    record['endpoint_time_difference_s'] -= float(value['base_command_time_s'])
                    record['within_descriptive_tolerance'] = (
                        abs(float(value['gain'])-1.) <= GAIN_TOLERANCE+1e-12
                        and float(value['orthogonal_m']) <= ORTHOGONAL_TOLERANCE_M+1e-12)
                records.append(record)
    fields = list(dict.fromkeys(key for record in records for key in record))
    with (output / 'paired_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def _fmt(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def _report(output, summary, aggregate, run_script):
    experiment = summary['experiment']
    models = list(aggregate['groups'])
    distance_mm = float(experiment['distance_m'])*1000
    lines = [
        '# 螺丝与源底座共同平移的 ACT / DP 配对实验', '',
        f"实验时间：{experiment.get('created_at', '未记录')}。螺丝和源底座共同沿 ±X、±Y 平移 **{distance_mm:g} mm**；目标底座、机械臂初始状态、相机和物体外观保持不变。",
        f"包含 {len(experiment['positions'])} 个初始布局、{len(summary['rows'])} 个平移配对。ACT seeds：{experiment.get('act_seeds')}；DP seeds：{experiment.get('diffusion_seeds')}。同一场景的不同 DP seeds 是采样重复，不是新布局。", '',
    ]
    for model in models:
        groups = aggregate['groups'][model]['all']['all']
        fk = groups['metrics']['baseline_first_closure']['statistics']['along_mm']['mean']
        actual = groups['realized_metrics']['baseline_first_closure']['statistics']['along_mm']['mean']
        independent = groups['metrics']['independent_first_closures']['statistics']['along_mm']['mean']
        lines.extend([f"**{MODEL_NAMES.get(model, model)}**：基准闭爪同 query 处，两个条件的预测 TCP 目标沿平移方向平均相差 **{_fmt(fk)} mm**，实际 TCP 位置平均相差 **{_fmt(actual)} mm**；按各自闭爪时刻比较，预测 TCP 目标平均相差 **{_fmt(independent)} mm**。参照物体平移为 **{distance_mm:g} mm**。", ''])
    gains = [aggregate['groups'][model]['all']['all']['metrics']['baseline_first_closure']['statistics']['gain']['mean']
             for model in models]
    if all(gain is not None and gain < 1-GAIN_TOLERANCE for gain in gains):
        lines.extend(['本批次所有已测模型的平均位置响应均低于预设的同步跟随容差。预测端已经出现响应不足，所以不能把全部偏差解释为执行器没有跟上；但本实验仍不能把原因唯一定位到内部空间表示。', ''])
    for model in models:
        stats = aggregate['groups'][model]['all']['all']['metrics']['baseline_first_closure']['statistics']
        lines.extend([f"{MODEL_NAMES.get(model, model)} 在该闭爪代理时刻的预测 TCP 到螺丝抓取参考点的平均 XY 误差：基准 **{_fmt(stats['baseline_grasp_xy_error_mm']['mean'])} mm**，平移后 **{_fmt(stats['shifted_grasp_xy_error_mm']['mean'])} mm**。这是目标几何误差，不是抓取成功率。", ''])
    lines.extend([
        '## 怎样读结果', '',
        '本实验比较“平移条件 TCP − 基准条件 TCP”在物体移动方向上的有符号投影。gain=1 表示末端目标同步移动完整距离，gain=0 表示该方向没有响应，负值表示反向响应。只看位移长度可能掩盖方向错误，所以同时报告正交位移和相对理想平移向量的残差。', '',
        '抓取阶段优先比较两个指标：① 基准第一次连续闭爪指令的同一个 query；② 两个条件各自第一次连续闭爪指令的 query。前者保持预测时间一致，后者允许抓取时序变化。闭爪仅为指令开度连续两步不大于阈值的代理，未证明夹住物体；缺失时不填零、不改用最后动作。', '',
        f"闭爪阈值为 {float(experiment['closure_proxy_width_m'])*1000:g} mm。完整 64 步 chunk 后段可能已进入拔出阶段，因此最后动作只作为补充。", '',
        '以下数值先在每个布局/方向内平均 DP seeds，再在布局内等权平均可用方向，最后等权平均布局。表中的样本数同时保留有效配对和有效布局；不同分组存在缺失或范围内/外包含方向不同时，不能视为相同样本总体。每个 DP seed 的基准闭爪 query 也可能不同。', '',
        f'预先固定的描述性容差为 **|gain−1|≤{GAIN_TOLERANCE:g} 且正交位移≤{ORTHOGONAL_TOLERANCE_M*1000:g} mm**。对于 30 mm 平移，即方向投影为 24–36 mm，同时侧向偏差不超过 5 mm。先逐配对判定再按上述布局权重汇总，避免先平均造成正负偏差抵消。表中比例以全部请求配对为分母，所选端点缺失不计满足；JSON 另外保留只基于有效端点的比例。同 query 指标只要求基准有闭爪，平移条件未闭爪时该时刻位置仍然可比较。该比例不等于抓取成功率，也不是显著性检验。', '',
    ])
    for source, heading in (('metrics', '预测关节目标经 FK 得到的 TCP'),
                            ('realized_metrics', '原始动作完整开环执行后的实际 TCP')):
        lines.extend([f'## {heading}', '',
                      '| 模型 | 分组 | 闭爪比较 | 有效配对 / 总数 | 有效布局 | 沿移动方向 mm | gain | 正交 mm | 残差 mm | 容差内比例 |',
                      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|'])
        for model in models:
            for scope in SCOPES:
                for endpoint in ENDPOINTS[:2]:
                    group = aggregate['groups'][model][scope]['all'][source][endpoint]
                    stats = group['statistics']
                    fraction = group['descriptive_tolerance']['all_requested_layout_weighted_fraction']
                    lines.append('| '+ ' | '.join([
                        MODEL_NAMES.get(model, model), SCOPE_NAMES[scope], ENDPOINT_NAMES[endpoint],
                        f"{group['available_rows']} / {group['paired_rows']}",
                        f"{group['layouts_available']} / {group['layouts_total']}",
                        *[_fmt(stats[field]['mean']) for field in ('along_mm', 'gain', 'orthogonal_mm', 'residual_mm')],
                        '—' if fraction is None else f'{fraction:.1%}',
                    ])+' |')
        lines.append('')
    lines.extend(['## 完整 chunk 终点与第 32 步补充比较', '',
                  '以下直接对应按顺序执行 chunk 后的端点差；共同初始 TCP 已消去，所以也是两条轨迹累计 Cartesian 净位移之差。它不是把绝对关节角相加。完整 chunk 最后一个目标可能处于抓取之后的阶段，应结合上面的闭爪指标解释。', '',
                  '| 模型 | 分组 | 端点 | FK 沿方向 mm | 实际沿方向 mm | FK gain | 实际 gain | FK / 实际有效配对 |',
                  '|---|---|---|---:|---:|---:|---:|---:|'])
    for model in models:
        for scope in SCOPES:
            groups = aggregate['groups'][model][scope]['all']
            for endpoint in ('query31', 'last'):
                fk = groups['metrics'][endpoint]
                actual = groups['realized_metrics'][endpoint]
                lines.append('| '+' | '.join([
                    MODEL_NAMES.get(model, model), SCOPE_NAMES[scope], ENDPOINT_NAMES[endpoint],
                    _fmt(fk['statistics']['along_mm']['mean']),
                    _fmt(actual['statistics']['along_mm']['mean']),
                    _fmt(fk['statistics']['gain']['mean']),
                    _fmt(actual['statistics']['gain']['mean']),
                    f"{fk['available_rows']} / {actual['available_rows']}",
                ])+' |')
    lines.extend(['', '## 方向差异与缺失情况', '',
                  '| 模型 | 方向 | 同 query 闭爪 FK mm | 同 query 闭爪实际 mm | 各自闭爪 FK mm | 各自闭爪实际 mm |',
                  '|---|---|---:|---:|---:|---:|'])
    for model in models:
        for direction in DIRECTIONS[1:]:
            groups = aggregate['groups'][model]['all'][direction]
            values = [_fmt(groups[source][endpoint]['statistics']['along_mm']['mean'])
                      for endpoint in ENDPOINTS[:2] for source in SOURCES]
            lines.append('| '+' | '.join([MODEL_NAMES.get(model, model), direction, *values])+' |')
    lines.append('')
    for model in models:
        counts = aggregate['closure'][model]['all']['all']
        replay = aggregate['replay'][model]
        lines.append(f"- {MODEL_NAMES.get(model, model)}：{counts['paired_rows']} 个平移配对中，基准闭爪缺失 {counts['baseline_missing']}，平移条件闭爪缺失 {counts['shifted_missing']}，两侧均有闭爪 {counts['both_present']}。这里基准会被四方向重复引用。")
        if replay['requested']:
            lines.append(f"  实际开环重放完整完成 {replay['completed']} / {replay['unique_predictions']} 条独立动作序列；不完整 {replay['incomplete']}，记录不可用 {replay['unavailable']}。完整执行仅表示执行完 chunk，不表示抓取或插入成功。")
        else:
            lines.append('  未请求实际物理重放。')
    lines.extend(['', '## 动作、时间和结论边界', '',
                  '- 模型输出 7 个绝对关节角和夹爪开度。逐个绝对目标做 FK；不能把关节角直接相加。连续 Cartesian 位移之和等于末端终点减初始点，配对比较时相同初始 TCP 消去。',
                  '- 本实验只输入初始 RGB-D/state，预测完整 64 个动作并以 10 Hz 开环执行，物理为 1 kHz；期间不重规划、不裁剪目标、不加入专家修正。常规 n32 在 3.2 秒后重新观察/预测，本实验的后 32 步不代表该闭环部署行为。',
                  '- FK 的 query k 是 k/10 秒下发的目标；实际 TCP 是执行该目标 100 ms 后的观测，即 (k+1)/10 秒。CSV 分别保存指令时刻与物理观测时刻。query 31 对应 3.1 秒目标、3.2 秒实际观测，query 63 对应 6.3 秒目标、6.4 秒实际观测。',
                  '- “训练范围内”只指平移后的源位置位于训练 source XY 的逐轴最小/最大值范围内；不保证该坐标见过、不保证源/目标组合在训练分布内，也不等于严格插值。范围外单独列出，不能把外推失效全部归为空间编码不足。',
                  '- 若抓取阶段预测位移持续明显小于物体位移，可作为策略对位置变化响应不足的证据；它不能单独定位内部空间表征。动作头、训练覆盖、时序、物体/底座视觉相关性都可能参与。若 FK 已充分跟随而实际执行不足，应进一步检查动力学跟踪、接触和时序。早期接近动作无需立即移动完整距离。',
                  '- 本次同时移动螺丝与源底座，无法区分模型对两者各自的视觉依赖；结果也不是闭环完整任务成功率。',
                  '- 可供下一轮因果区分的实验：在相同训练/测试划分、动作表示与控制设置下，对比原输入和加入真实物体位姿的模型，再分别比较 FK 与实际 TCP。该对照用于区分位置输入/编码问题和动作生成/执行问题；本次没有训练或验证点云、位姿估计方案。', '',
                  '## 文件与复现', '',
                  '- [逐配对所有端点 CSV](paired_metrics.csv)：包含缺失标记、query、时间、方向投影、正交误差和残差。',
                  '- [描述性统计 JSON](aggregate.json)：所有端点，按模型/方向/训练坐标范围分组；mean/median/min/max 对应等权布局均值，raw_row_range 另保留原始配对极值；另含平均 |gain−1| 和配对端点时间差。',
                  '- [完整实验配置及原始指标](summary.json)、[原始预测和物理轨迹](data/)、[配对场景与图像](scenes/)。',
                  '- [预测响应曲线](plots/translation_vs_horizon.png) / [PDF](plots/translation_vs_horizon.pdf)；[端点 gain 图](plots/endpoint_translation_gain.png) / [PDF](plots/endpoint_translation_gain.pdf)。', '',
                  '- [预测与实际执行对比](plots/predicted_vs_executed.png) / [PDF](plots/predicted_vs_executed.pdf)。', '',
                  f"权重与场景配置 SHA256 前后检查：**{summary.get('protected_unchanged')}**。完整哈希记录在 summary.json 的 protected_before / protected_after。", ''])
    if (output/'audit.json').is_file():
        lines.extend(['[完整性与复现审计](audit.json)。审计完成不等于物理抓取成功。', ''])
    for model in models:
        policy = summary['models'][model]['policy']
        checkpoint = policy['checkpoint']
        model_hash = summary.get('protected_before', {}).get(str(Path(checkpoint)/'model.safetensors'), '未记录')
        lines.extend([f'**{MODEL_NAMES.get(model, model)}**', '',
                      f'- checkpoint：`{checkpoint}`',
                      f'- model SHA256：`{model_hash}`',
                      f"- config SHA256：`{policy.get('config_sha256', '未记录')}`",
                      f"- device：`{policy.get('device')}`；PyTorch：`{policy.get('torch_version')}`；LeRobot：`{policy.get('lerobot_version')}`。",
                      f"- depth_unit：`{policy.get('depth_unit')}`；chunk：`{policy.get('chunk_size')}`；n_obs_steps：`{policy.get('n_obs_steps')}`；去噪步数：`{policy.get('num_inference_steps', '不适用')}`。", ''])
    if run_script is not None:
        script = Path(run_script).expanduser().resolve()
        lines.extend([f'本次保存的启动脚本：`{script}`。重新运行须替换输出目录以保留已有结果。', '',
                      '```bash', script.read_text(encoding='utf-8').rstrip(), '```', ''])
    else:
        lines.extend(['本目录未找到 run.sh；精确运行参数见 experiment.json，不能据此声称保存了原始 shell 命令。', ''])
    lines.extend(['重新汇总和绘图：', '', '```bash',
                  f'/usr/bin/python3 -m examples.insert_plug.policy.summarize_translation_probe --output {output}',
                  f'/usr/bin/python3 -m examples.insert_plug.policy.plot_translation_probe --output {output}',
                  '```', ''])
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def summarize_results(output: Path, run_script=None):
    output = Path(output).expanduser().resolve()
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    if summary.get('complete') is not True:
        raise ValueError('Only summarize a completed translation experiment')
    rows = summary['rows']
    if not rows or any(row['condition'] != 'peg_and_holder' for row in rows):
        raise ValueError('Expected nonempty screw-and-source-holder translation rows')
    if any(type(row['within_training_source_bounds']) is not bool for row in rows):
        raise ValueError('Training source coordinate bounds must be explicitly classified')
    models = list(dict.fromkeys(row['model'] for row in rows))
    groups, closure = {}, {}
    for model in models:
        groups[model], closure[model] = {}, {}
        for scope in SCOPES:
            groups[model][scope], closure[model][scope] = {}, {}
            for direction in DIRECTIONS:
                selected = [row for row in rows if _selected(row, model, scope, direction)]
                groups[model][scope][direction] = {
                    source: {endpoint: _group(selected, source, endpoint) for endpoint in ENDPOINTS}
                    for source in SOURCES}
                closure[model][scope][direction] = _closure_counts(selected)
    experiment = summary['experiment']
    aggregate = dict(
        schema_version=1, experiment=experiment,
        aggregation='Mean over seeds within layout/direction, mean over available directions within layout, equal-weight summary over available layouts. No inferential tests.',
        statistics_semantics='mean/median/min/max are over layout-level means; raw_row_range is unaggregated paired-row range.',
        descriptive_threshold=dict(gain_absolute_deviation_max=GAIN_TOLERANCE,
                                   orthogonal_m_max=ORTHOGONAL_TOLERANCE_M,
                                   rule='Both criteria required; tolerance is descriptive, not grasp success or statistical significance.'),
        groups=groups, closure=closure,
        replay=_replays(output, rows, experiment.get('replay_full_chunk', False), experiment['full_chunk_steps']),
        protected_unchanged=summary.get('protected_unchanged'),
    )
    aggregate['csv_rows'] = _csv(output, rows)
    (output/'aggregate.json').write_text(json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    if run_script is None and (output/'run.sh').is_file():
        run_script = output/'run.sh'
    _report(output, summary, aggregate, run_script)
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-script', type=Path)
    args = parser.parse_args(argv)
    result = summarize_results(args.output, args.run_script)
    print(json.dumps(dict(output=str(args.output), csv_rows=result['csv_rows'],
                          files=['aggregate.json', 'paired_metrics.csv', 'report.md']), indent=2))


if __name__ == '__main__':
    main()
