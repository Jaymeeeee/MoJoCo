"""Export a concise Chinese report and per-direction CSV from verified probes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(output):
    summary = json.loads((output/'summary.json').read_text())
    if summary['status'] != 'complete' or not summary['baseline_gate_passed']:
        raise ValueError('All gated experiments must complete before summarizing')
    labels = {'act': 'ACT', 'diffusion': 'DP'}
    directions = {'x_plus': '+X', 'x_minus': '-X', 'y_plus': '+Y', 'y_minus': '-Y'}
    lines = ['已验证原点抓取的 RGB-D ACT / DP 3 cm 平移实验', '',
        '本组共 10 条仿真：两个模型各一次原点，加四个平移条件。共同基准为 pos2，seed=20。',
        'pos2 根据此前两模型各 10/10 的成功记录预先选定；本次两个原点再次通过实际抓起验证后才运行平移。',
        '螺丝与源底座一起平移，目标底座固定。机械臂初态、相机、外观、动力学和推理参数保持一致。',
        '模型保留 64 步预测范围，每次执行 32 个绝对关节目标再重规划；控制 10 Hz、物理 1 kHz。',
        '每条固定运行 12.8 秒，完整随机流逐次重规划配对。没有叠加绝对关节角或重叠的预测 chunk。', '',
        '原点资格：左右夹指连续接触至少 200 ms，螺丝尖端相对 TCP 的位置漂移不超过 0.2 mm；',
        '另要求尖端高于源底座口沿至少 5 mm，并同时保持上述稳定夹持至少 200 ms。', '']
    for model, episodes in summary['episodes'].items():
        q = episodes['baseline']['qualification']
        lines.append(f"{labels[model]} 原点：抓取确认 {q['grasp_event']['confirmed_time_s']:.3f} s，"
                     f"提离确认 {q['lift_event']['confirmed_time_s']:.3f} s，"
                     f"提离窗口最大相对漂移 {q['lift_event']['max_relative_drift_m']*1000:.3f} mm。")
    lines += ['', '以下是原点确认抓取所在控制周期末（两个模型均为 6.4 s）的配对比较：',
              '数值为“平移条件 TCP - 原点 TCP”在物体平移方向上的投影，理想值 +30 mm。',
              '负数表示平移条件的目标相对原点目标向反方向偏；不等同于机械臂整段运动朝反方向。',
              'FK 表示模型绝对关节目标的正运动学位置；实际表示 MuJoCo 执行后的 TCP。', '',
              '模型  方向  FK响应mm  实际响应mm  正交偏差mm(实际)  到源XY误差mm(实际)  窗口内稳定提起  源坐标包围盒']
    rows = []
    for model, pairs in summary['pairs'].items():
        for direction, pair in pairs.items():
            result = summary['episodes'][model][direction]
            endpoint = pair['baseline_confirmed_grasp']
            qualified = result['qualification']['passed']
            independent = pair['independent_confirmed_grasps']
            row = dict(model=labels[model], direction=directions[direction],
                       fk_along_mm=endpoint['fk']['along_m']*1000,
                       actual_along_mm=endpoint['actual']['along_m']*1000,
                       actual_orthogonal_mm=endpoint['actual']['orthogonal_m']*1000,
                       actual_vector_residual_mm=endpoint['actual']['residual_m']*1000,
                       actual_xy_error_mm=pair['matched_time_shifted_actual_xy_error_mm'],
                       lifted_within_12_8s=qualified,
                       within_training_source_bounds=result['within_training_source_bounds'],
                       independent_fk_along_mm=independent['fk']['along_m']*1000 if independent else None,
                       independent_actual_along_mm=independent['actual']['along_m']*1000 if independent else None)
            rows.append(row)
            lines.append(f"{row['model']:4} {row['direction']:3} {row['fk_along_mm']:9.2f} {row['actual_along_mm']:11.2f} "
                         f"{row['actual_orthogonal_mm']:17.2f} {row['actual_xy_error_mm']:20.2f} "
                         f"{'是' if qualified else '未验证':10} {'内' if row['within_training_source_bounds'] else '外'}")
    lines += ['', '各自的抓取时刻（只有两边都通过实际提起才比较，避免把空夹或接触误当成功抓取）：']
    for row in rows:
        if row['lifted_within_12_8s']:
            lines.append(f"{row['model']} {row['direction']}：FK响应 {row['independent_fk_along_mm']:.2f} mm，"
                         f"实际响应 {row['independent_actual_along_mm']:.2f} mm。")
    if not any(row['lifted_within_12_8s'] for row in rows):
        lines.append('四方向平移均未在本窗口内通过稳定提起，因此没有可比较的成功独立抓取终点。')
    lines += ['', '如何解释这三张图：',
        '1. baseline_grasp_qualification.png：证明两个原点确实能夹住并提起，消除原实验基准不合格的混淆。',
        '2. translation_vs_time.png：展示完整时间响应，区分晚到、欠响应和执行偏差。30 mm 参考只适用于对应抓取阶段；转移目标底座未移动。',
        '3. grasp_translation_response.png：在原点抓取周期比较四方向的预测与实际响应，并标注平移条件是否提起。', '',
        '结论边界：',
        '原点能成功而平移后在抓取阶段欠响应，可支持“模型对位置变化适应不足”的诊断。',
        '但同一时刻的位移不足并不单独证明最终抓取目标错误，须结合整段曲线、独立抓取事件和到源位置误差。',
        '每条仅观察 12.8 s；未通过表示窗口内未验证稳定提起，不表示无限延长后必然失败。',
        '+X、+Y 超出训练源坐标的包围盒；-X、-Y 在盒内也不等于严格插值。',
        '这是一处已知可成功位置、单一种子的小样本因果对照，不是总体成功率或全部位置的统计结论。',
        '后续状态反馈也参与闭环响应；不能仅凭本实验唯一归因于编码器内部空间表示，或认定点云/位姿输入一定有效。',
        '旧实验仍保留，只能解读为初始 chunk 的位置敏感性；本组补上了成功原点和实际提起的条件。', '',
        '原始数据：protocol.json、summary.json、episodes/<模型>/<方向>/trace.npz、result.json、rollout.mp4。']
    (output/'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (output/'paired_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for row in summarize(args.output.resolve()):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
