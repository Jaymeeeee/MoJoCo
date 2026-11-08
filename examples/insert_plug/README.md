# insert_plug 插销任务

FR3 配合 Robotiq 2F-85，从一个放置桶夹起六角头螺栓，沿孔轴拔出、搬运、对准并浅插另一个放置桶，再松爪让零件靠重力落座。

机械臂和夹爪来自固定版本的 MuJoCo Menagerie，保留各自许可证和校验清单。物件外观使用用户提供的 `assets/放置桶.STL` 与 `assets/螺栓.STL`。机器人使用理想重力补偿，零件自由运动并保留重力、摩擦和接触；质量和摩擦系数仍是可调整的仿真假设。

## 文件位置

| 内容 | 工作区内路径 |
|---|---|
| 场景与物理参数 | `config/insert_plug.yaml` |
| 10 组已验证的 2 cm 网格位置 | `config/insert_plug_positions.yaml` |
| FR3 模型与许可证 | `assets/robots/franka_fr3/` |
| Robotiq 模型与许可证 | `assets/grippers/robotiq_2f85/` |
| 原始零件网格 | `assets/放置桶.STL`、`assets/螺栓.STL` |
| 从 STL 派生的碰撞几何 | `assets/objects/insert_plug/` |
| 场景构建 | `examples/insert_plug/scene.py` |
| 当前配置生成的完整场景 | `examples/insert_plug/scene.xml` |
| 本任务物件装配 | `examples/insert_plug/objects.py` |
| 通用 Robotiq 装配与开合控制 | `src/mujoco_tools/robotiq.py` |
| 轨迹规划、执行检查与仿真接口 | `planning.py`、`expert.py`、`execution_guard.py`、`simulation.py` |
| 本场景数据格式与视频回放 | `dataset.py`、`replay.py` |
| 采集数据（每条 HDF5、YAML、XML、第三视角 MP4） | `outputs/datasets/insert_plug/` |
| 合并后的 LeRobot v3 数据集 | `outputs/datasets/insert_plug/lerobot/` |
| LeRobot 转换入口与 ACT 输入配置 | `scripts/export_insert_plug_lerobot.py`、`config/act_insert_plug_features.json` |
| 按需生成的图片、视频、计划导出 | `outputs/insert_plug/` |

## 运行

以下命令均在工作区顶层执行：

```bash
conda activate mujoco
cd ~/Projects/mujoco

# 根据当前配置重新生成 examples/insert_plug/scene.xml
python -m examples.insert_plug build

# 只看静态场景，不执行动作、不采集、不保存输出
python -m examples.insert_plug view

# 查看指定相机：head 对应固定第三视角，也可指定 wrist
python -m examples.insert_plug view --camera head

# 保存初始场景 RGB、原始深度与深度预览，不推进物理或采集轨迹
python -m examples.insert_plug snapshot --backend egl

# 只规划并检查完整名义轨迹，导出 JSON/NPZ，不执行动作或采集图像
python -m examples.insert_plug plan --output outputs/insert_plug/plans

# 执行动作并采集，显示实时窗口
python -m examples.insert_plug demo --backend glfw --viewer --realtime

# 先规划检查，再执行并保存 512×512 双相机观测与视频
python -m examples.insert_plug demo --backend egl --width 512 --output outputs/datasets/insert_plug/under20_demo --video

# 无图像物理检查；产物属于可自动清理的临时结果
python -m examples.insert_plug demo --no-images

# 按当前固定底座位置采集 10 条，每条默认附带第三视角 MP4
python -m examples.insert_plug collect --backend egl --episodes 10 --randomize-mm 0 \
  --output outputs/datasets/insert_plug/under20_fixed_v1

# 需要随机孔位时显式设置随机范围
python -m examples.insert_plug collect --backend egl --episodes 10 --seed 200 --randomize-mm 2
```

`--width 128` 可降低两路相机分辨率，默认各 256×256。记录频率由 `config/insert_plug.yaml` 的 `recording.sample_hz` 或 `--sample-hz` 设置，默认 10 Hz。`--control-hz` 独立设置专家控制频率，默认也为 10 Hz，并须整除物理频率；默认物理步长为 1 ms。每次控制指令在 100 个物理步内保持恒定，即 100 ms。没有桌面显示的环境不要使用 `--viewer`；GPU 不可用时可指定 `--backend osmesa`。

```bash
python -m examples.insert_plug validate outputs/datasets/insert_plug/pos0/episode_000000/episode_000000.h5
python -m examples.insert_plug preview outputs/datasets/insert_plug/pos0/episode_000000/episode_000000.h5 --output outputs/insert_plug/replay.mp4
```

每条记录默认生成同名 `.mp4`，只包含第三视角 `head` RGB，以 HDF5 的采样频率播放（默认 10 fps），不叠字、不缩放，帧数与 HDF5 一致。带 `--video` 的上述命令还会额外将双视角视频另存到 `outputs/insert_plug/previews/under20_demo/preview.mp4`。额外视频左侧为 Head (third-person)，右侧为 Wrist；独立黑色页眉显示相机名称、记录的仿真时间和当前阶段。已有 episode 和视频不会覆盖，也不会随当前配置自动更新；重复执行这条命令请改用新的输出目录。

省略 `--output` 时，`demo` 与 `collect` 都保存到 `outputs/datasets/insert_plug/run_<时间>/`；指定 `--output` 时使用该批次目录。每个 episode 必须放在独立的 `episode_<seed>/` 子目录中，内部保留同名 `.h5`、`.yaml`、`.xml`、`.mp4` 四个文件。历史 15 Hz 固定位置测试批保留在 `outputs/datasets/insert_plug/test/`，结构如下：

```text
test/
├── episode_000000/
│   ├── episode_000000.h5
│   ├── episode_000000.yaml
│   ├── episode_000000.xml
│   └── episode_000000.mp4
├── episode_000001/
│   └── ...
└── episode_000009/
    └── ...
```

这个历史测试批的 10 条均已成功完成并通过校验，每条 384 帧、15 Hz，保持原样。新采集默认使用 10 Hz，应保存到新的批次目录或使用不同 seed。显式请求的额外双视角视频放到 `outputs/insert_plug/previews/<运行目录名>/`。根目录 `data/peg_transfer/` 的旧记录已清理，旧 `outputs/peg_transfer/` 历史记录仍保留；旧资产目录名通过符号链接兼容历史场景。

`examples/insert_plug/scene.xml` 是当前配置生成的场景；修改配置后可用 `build` 更新。采集记录中的 `episode_<seed>.xml` 是该次运行的场景快照，配套 YAML 保存本次参数，包括随机化的底座位置。HDF5 中 `model_path` 与 `execution_result.model` 保存该 XML 的相对文件名，以 HDF5 所在的 episode 目录为基准，便于整体移动记录。XML 仍引用共享 `assets/`，重建仿真时需要这些资源。`view` 和静态预览按当前配置构建场景，不会改写历史 episode。

## 网格位置批量采集

批量布局在 `config/insert_plug_positions.yaml` 中定义，包含基础配置路径及 SHA256、2 cm 网格间距、10 组 `source_xy`/`target_xy` 和整数偏移 `source_steps`/`target_steps`。当前选择范围为相对于基准位置的每轴 ±4 cm，各组源底座和目标底座的位置都分别唯一；每个位置内部固定采集 10 条，不加连续随机扰动。

位置筛选读取实际编译后的 STL 视觉网格全部顶点，检查两底座和初始、最终零件是否落在第三视角视锥内，并保留至少 5 px 边缘余量。每组还完成规划预检和完整物理预执行，清单中保留简洁检查结果。清单按当前专家配置验证并更新基础配置 SHA256，十组底座坐标没有变化。默认采集保留抓稳、到位和任务完成条件；普通接触由 MuJoCo 处理，不运行附加接触阈值中止。

```bash
# 10 组位置，每组 10 条，控制/记录均为 10 Hz、双路 RGB-D、第三视角 MP4
python scripts/collect_insert_plug_positions.py --workers 3 \
  --output outputs/datasets/insert_plug/under20_v1

# 可单独选择某组；重复运行会校验并跳过同配置的成功记录
python scripts/collect_insert_plug_positions.py --position pos0 --workers 1 \
  --output outputs/datasets/insert_plug/under20_v1

# 整批完成后，检查位置、频率、深度、动作时序和视频
python scripts/audit_insert_plug_batch.py --positions-config config/insert_plug_positions.yaml \
  --root outputs/datasets/insert_plug/under20_v1
```

上述命令为后续平滑专家采集预留独立的 `under20_v1/` 目录，不会覆盖现有训练数据。当前保留的 100 条来自先前专家，采用 10 Hz 控制、10 Hz 记录、1 kHz 物理，并曾替换更早的 50 Hz 控制／10 Hz 记录批次。该历史批次 **100/100 条成功，共 38,410 帧、400 个文件，约 13.64 GB**，采集时通过整批独立审计和保存动作物理回放；初始和最终视觉网格的画面边缘最小余量为 51.694 px。这些数字不代表当前平滑专家已经重新采集 100 条。历史记录结构为：

```text
outputs/datasets/insert_plug/
├── pos0/
│   ├── episode_000000/
│   │   ├── episode_000000.h5
│   │   ├── episode_000000.yaml
│   │   ├── episode_000000.xml
│   │   └── episode_000000.mp4
│   └── ... episode_000009/
├── pos1/
│   └── episode_000010/ ... episode_000019/
├── ...
└── pos9/
    └── episode_000090/ ... episode_000099/
```

`pos{i}` 使用全局编号 `i*10` 至 `i*10+9`。新命令使用相同子目录结构，但位于 `outputs/datasets/insert_plug/under20_v1/` 下。基础配置和批量清单的控制、记录频率均为 10 Hz，批量采集开启两路深度；原 `test/` 中的历史 15 Hz 记录保持原样。每条 HDF5 和 YAML 都保存本次实际配置与所属位置；每条动作在采样时生效，并保持至 100 ms 后的下一次观测。

安全续采仅接受四个文件齐全、HDF5 校验成功且 YAML 配置完全一致的已有记录。遇到缺失文件、失败轨迹或配置不同会停止并保留现有内容。基础配置 SHA256 若与清单不一致，也会拒绝启动，避免使用过时的位置预检结果。

需要根据修改后的基础配置生成新布局时，使用新的清单和输出目录：

```bash
python -m examples.insert_plug.layouts --config config/insert_plug.yaml --output config/insert_plug_positions_new.yaml
python scripts/collect_insert_plug_positions.py --positions-config config/insert_plug_positions_new.yaml --output outputs/datasets/insert_plug/new_positions --workers 3
```

布局生成器默认不覆盖已有清单；`--position` 可重复指定以选择多个位置。后续新批次完成后，可通过下述独立转换步骤合并为 LeRobotDataset，原始四类文件仍保留。

## 10 Hz 动作物理回放检查

`scripts/check_insert_plug_action_replay.py` 默认逐条独立执行 `pos0` 至 `pos9` 的全部 100 条保存动作，只读原始记录，报告默认写入 `outputs/_checks/insert_plug_action_replay_<时间>/report.json`。在工作区顶层运行：

```bash
conda activate mujoco
env -u PYTHONPATH python scripts/check_insert_plug_action_replay.py

# 可用 --episode 选择单条，重复该参数可选择多条
env -u PYTHONPATH python scripts/check_insert_plug_action_replay.py \
  --episode outputs/datasets/insert_plug/pos0/episode_000000/episode_000000.h5 \
  --workers 1
```

`--action-dtype` 默认为 `float32`，与 LeRobot 导出动作一致；指定 `float64` 可检查原始 HDF5 精度。当前新记录在实际下发前已量化为 float32，因此这两种读取方式的控制数值相同。每条从配套场景的正常 reset 开始，读取记录内的控制频率：当前 10 Hz 控制／10 Hz 记录的每个动作只执行一次 100 ms 的 `Simulation.step()`；旧的 50 Hz 控制／10 Hz 记录则将每个目标重复执行五次 20 ms 控制步，并将末动作截断到原执行时长。两种情况下均由执行器和 1 kHz 物理推进，不恢复中间状态、不调用专家。这是保存动作的独立物理验证，不属于策略评估或录像回放。

报告中的 `task_success` 要求完整执行、最终落座条件通过，并在最后 250 ms 的 251 个物理时刻全部稳定；稳定性指标在独立 `MjData` 副本上刷新计算。`contact_clean` 表示全程无接触检查违规，接触监测不依赖原阶段标签，也不中止回放；总 `success` 要求两者同时成立。

保留的先前专家 100 条记录均通过独立物理重放：`task_success`、`contact_clean` 和总 `success` 全部为真；相同时间的关节及插销位置与原记录最大偏差均为零，最后 250 ms 的 251 个物理时刻全部满足成功条件。这是历史保存动作的执行一致性检查，不是平滑专家新数据的整批结果，也不是训练后 ACT 的成功率。

更早的结果：2026-11-07 对随后被重采替换的旧批次（50 Hz 专家控制、10 Hz 记录）检查时，100 次稀疏动作物理回放中有 90 次最终完成插入，`pos6` 的十次均未完全落座；100 次都出现超过原阈值的孔壁／孔口接触，严格总通过数为零。每组十条保存的动作相同，共覆盖十种轨迹。原 50 Hz 专家参考运行能复现成功终态且接触零违规，`pos0`、`pos6` 改用原始 float64 动作也未消除稀疏回放问题。该历史结果说明稀疏保存动作不能恢复原来的中间控制更新，**不是保留的 10 Hz 控制／10 Hz 记录批次的回放结果，也不是 ACT 成功率**。未来新批次仍应在完成采集后独立回放验证。

## 规划与执行

两个放置桶固定在世界坐标系中，没有 joint、freejoint 或 mocap，不会被碰撞推走。平面位置只由 `config/insert_plug.yaml` 中的 `board.source_xy`（橙色源孔，默认 `[0.45, -0.07]`）和 `board.target_xy`（绿色目标孔，默认 `[0.45, 0.07]`）定义，单位米。调整布局时修改这两个配置项，规划器从生成场景的 site 读取位置；不需要另传一组目标坐标。`collect` 默认 `--randomize-mm 0`，同一 episode 中位置始终固定；显式设置非零范围才会在创建各 episode 时改变布局。

`plan` 在独立的 `MjData` 中生成任务约束下的平滑笛卡尔轨迹与姿态插值，以上一采样点的关节解为种子逐点求解 IK，并按关节速度、加速度限制调整时长。默认提起使用余弦速度加减速与匀速段；抓取点、目标上方路点的连接采用指定端点速度、零端点加速度的五次 Hermite 曲线，其余段使用平滑五次曲线。重定时检查各段的笛卡尔及关节运动界限。默认 `task.planning.strict_collision_checks=false`，仅做 IK、运动界限及任务几何检查，不按采样碰撞阈值拒绝轨迹。执行时保留原碰撞体、摩擦和接触求解，不能把规划成功理解为无障碍保证。显式开启该选项才运行旧的密集碰撞诊断；它也不是通用 RRT／全局避障搜索或连续无碰撞证明。

独立 `plan` 命令保存 `nominal_trajectory.json` 和 `nominal_trajectory.npz`：JSON 包含阶段位姿、时长、边界和检查信息；NPZ 保存时间、关节目标、夹爪开口、TCP 位姿、阶段及动作数组。规划过程不推进实际仿真，不生成采集 HDF5。

`demo` 和 `collect` 自动先对完整名义任务预检，成功后才开始执行和打开数据写入器。执行分为以下步骤：

1. 按名义轨迹完成初始稳定、接近、下降与夹持；通过抓取点上方时保持向下速度，不停稳后再下降。
2. 双指夹持稳定后，重新测量零件相对 TCP 的完整 SE(3) 变换，包括位置和旋转；据此重规划并检查拔出、搬运及后续轨迹。
3. 到目标上方时再次测量实际夹持变换，后续对准命令从上一条目标的位姿和速度接续，避免从滞后的实际位姿重新零速起步。搬运、对准和浅插连续向下衔接，默认预插入间距为 15 mm。进入浅插前即时核对孔轴匹配；默认连续模式若不满足则报告失败，不执行停住再纠偏的循环。显式关闭 `continuous_hover` 才保留旧的最多两次额外对准修正。
4. 通过插入条件后浅插、检查深度、松爪、撤离，最后检查自由落座稳定性。

当前默认专家连续通过抓取点、放置点上方，降低搬运高度，并按完整任务少于 20 秒调整速度与阶段时长：

| 参数 | 默认行为 |
|---|---|
| `task.approach_height` | 抓取点上方 0.04 m；这是相对抓取点的高度 |
| `task.transport_height` | TCP 离桌面 0.10 m；这是世界坐标绝对高度，没有额外隐藏的 0.07 m 提起下限 |
| `task.extraction_profile`、`extraction_duration`、`extraction_ramp_fraction` | `cosine_velocity`、4 秒、0.3；两端加减速各占 30%，中间匀速 |
| `task.descent_duration`、`transfer_duration`、`alignment_duration`、`insertion_duration` | 下降抓取 1.0 秒、搬运 1.5 秒、对准 1.2 秒、浅插 2.5 秒；运动界限可要求延长 |
| `task.planning.max_joint_velocity`、`max_joint_acceleration` | 关节速度上限 1.4 rad/s、加速度上限 4 rad/s² |
| `task.planning.max_cartesian_velocity`、`max_cartesian_acceleration` | 笛卡尔速度上限 0.18 m/s、加速度上限 0.8 m/s² |
| `task.planning.approach_max_cartesian_velocity` | 仅空载接近段允许 0.24 m/s，持物搬运仍受上述默认上限约束 |
| `task.planning.continuous_hover` | 开启；抓取上方、目标上方和预插入边界保持非零向下速度，不附加停稳等待 |
| `task.planning.grasp_approach_speed`、`place_approach_speed` | 上方路点向下速度分别为 0.02 m/s、0.008 m/s |
| `task.planning.adaptive_grasp_settle`、`grasp_settle_seconds` | 开启条件等待，0.8 秒为最长额外等待；双指稳定接触至少 200 ms 后继续 |
| `task.planning.tracking_stable_seconds` | 实际抓取和浅插终点稳定观察跨度为 0.1 秒，计入已有稳定观测 |
| `task.planning.continuous_free_space` | 提起和撤离不追加精确停稳等待；实际净空和最终完成条件保留 |
| `task.planning.alignment_correction_duration` | 旧的非连续模式中，小于 2 mm 的重测修正最短 0.4 秒；默认连续模式不使用停住纠偏 |
| `task.planning.insertion_depth` | 浅插 12 mm 后松爪 |
| `task.planning.retract_height` | 释放后从浅插 TCP 位置上移 0.05 m |
| `task.release_duration`、`task.settling_seconds` | 松爪 1 秒，末尾检查 0.5 秒 |
| `task.planning.strict_collision_checks` | 默认关闭附加碰撞阈值预检及执行中止；真实接触仍由 MuJoCo 求解 |

仍采用真实 10 Hz 控制，每个目标恒定保持 100 ms；没有插入隐含的高频目标。抓稳和插入精度条件保留；默认不创建 `MotionGuard`、不扫描附加接触阈值。计算重规划期间不推进仿真，也不增加记录中的等待帧。

当前默认在原十个固定布局各验证一次，**10/10 完成**，共 1,899 个控制步，执行时长 **18.6–19.7 秒、平均 18.99 秒**，全部少于 20 秒。十个布局的 `alignment_corrections` 均为 0，approach、transfer、align 段均没有重复保持指令。pos0 抓取上方交接前后实际速度约 22–24 mm/s，目标上方及预插入处保持约 8 mm/s 的连续下移。持物 TCP 峰值约 0.104 m，搬运时插销尖端距孔口约 0.026 m。见[当前验证报告](../../outputs/_checks/insert_plug_under20/summary.json)与[双视角示例](../../outputs/insert_plug/previews/under20.mp4)。这是已验证布局上的时长，改变位置、物体或参数后应重新验证。

默认场景 YAML 的固定底座位置另完成 **18.9 秒、189 帧、10 Hz** 的真实双路 RGB-D 录制和第三视角 MP4 审计。独立执行保存的动作、每条恒定保持 100 ms 后，任务成功，关节与插销状态相对记录的最大偏差均为零；双视角视频完整解码 189 帧，82 项回归测试通过。证据见[录制审计](../../outputs/_checks/insert_plug_under20/recording_validation.json)与[保存动作回放](../../outputs/_checks/insert_plug_under20/recorded_action_replay.json)。一次性原始试录文件清理，仅保留报告和视频。

上一版连续路点专家平均 24.42 秒，报告与视频保留在 [insert_plug_continuous_hover](../../outputs/_checks/insert_plug_continuous_hover/summary.json) 和 [continuous_hover.mp4](../../outputs/insert_plug/previews/continuous_hover.mp4)。再前一版平滑专家平均 30.3 秒，保留训练批次平均 38.41 秒。旧平滑版的[严格模式报告](../../outputs/_checks/insert_plug_smooth_expert/summary.json)、[关闭附加接触中止的报告](../../outputs/_checks/insert_plug_smooth_expert/contact_response/summary.json)及 [smooth_expert.mp4](../../outputs/insert_plug/previews/smooth_expert.mp4)均为历史对照；旧版 313 帧 RGB-D 试录、精确动作回放和 75 项回归测试也属于当时的验证。当前更新未改写已有 100 条数据、LeRobot 数据、checkpoint 或策略测试，不代表已重训 ACT 或提升模型成功率。抓稳、对准、插入和最终稳定要求均保留；默认接触诊断仍关闭，相关字段为 disabled/null，不声明“零违规接触”。

采集过程中仍会完成上述规划和任务条件检查，但默认不导出 `episode_<seed>_plan/`、结果 JSON 或汇总 JSON。阶段检查、执行结果和是否启用接触诊断存入 HDF5 元数据 `execution_result`，最终误差另在 `final_metrics` 中记录；控制台继续输出运行结果。实际经过的等待、修正和动作以 HDF5 的时间戳与阶段为准。需要检查独立的名义计划数组时，可显式运行 `plan` 命令导出到 `outputs/insert_plug/plans/`。

执行器在阶段间检查实际状态，默认条件如下：

| 检查 | 进入下一阶段的条件 |
|---|---|
| 实际抓取与浅插终点 | 下降抓取及浅插结束仍要求 TCP 位置误差不超过 50 μm、姿态误差不超过 0.5 mrad、最大关节速度不超过 0.01 rad/s；连续稳定观察跨度至少 100 ms，已有稳定观测可计入 |
| 上方路点与自由空间 | 抓取上方、搬运到目标及预插入边界连续通过，不等待零速停稳；提起后插销尖端仍须实际高于两孔口至少 15 mm，撤离后检查最终成功 |
| 夹持稳定 | 两侧指垫连续接触至少 200 ms，期间零件相对 TCP 的位置漂移不超过 0.2 mm |
| 插入几何 | 在预期落座高度与孔口高度检查整段杆轴；默认径向间隙 0.5 mm 减去 0.15 mm 余量，轴线偏差须小于 0.35 mm，零件倾角不超过 0.003 rad |
| 松爪与撤离 | 达到配置的浅插深度（默认 12 mm）后才能松爪；确认指垫脱离且开口足够后才能撤离 |

默认不因插销碰到孔壁、孔口或夹爪接触而中止；这些接触由 MuJoCo 自身的动力学处理。`task.planning.strict_collision_checks=true` 仅用于显式复现旧严格诊断，会启用逐物理步 `MotionGuard` 和离线碰撞阈值检查；默认无需开启。到位、实际夹持或插入条件失败仍可使专家停止当前任务，非法动作和数值异常也不会被忽略。碰撞掩码、约束和零件自由度不因关闭附加诊断而改变。

## 物理与控制

配置采用 SI 单位，相机 `fovy` 按 MuJoCo 约定使用角度。STL 按毫米转换为米，并把原始 +Y 轴旋转到世界 +Z。当前文件的杆径约 10 mm、孔径约 11 mm、杆长 30 mm；六角头对边 24 mm、对角约 27.71 mm、高 7.5 mm。放置桶高 40 mm、底厚 5 mm、孔深 35 mm。因此螺栓最终由头部支撑在桶口，杆尖高于内底约 5 mm，成功判定按这个落座高度执行。

2F-85 保留连杆、闭环约束和单个腱驱动器。策略动作中的开口仍以米表示，控制适配器将其转换为原模型的驱动指令；观测开口从两侧指垫位置测量。固定 TCP 按 24 mm 夹持宽度标定，专家旋转夹爪以夹持六角头的两侧平面。

运行时全部动作通过关节位置执行器和 `mj_step` 实现，零件不会焊接到夹爪或被瞬移。控制目标每 100 ms 更新一次，100 个 1 ms 物理子步内保持恒定。成功判定同时检查杆尖和孔口处轴线位置、落座高度、接触穿透、速度、夹爪撤离，以及至少 250 ms 的稳定性。专家使用带时间戳的连续成功观察计算跨度；10 Hz 下至少四个观察点才能覆盖 300 ms，当前预执行的最终稳定观察跨度为 400 ms。

`seating_error_m` 是相对于预期落座位置的误差；`floor_error_m` 始终是杆尖与实际内底的距离，本 STL 螺栓正确落座时约为 5 mm。

## 数据

新记录采用 `insert-plug-v3`，区分独立的记录频率与专家控制频率。第三视角的对外名称统一为 `head`，物理场景中的相机名称仍为 `third_person`；`view --camera head` 和原来的 `view --camera third_person` 均可用。旧 `peg-transfer-v1/v2` HDF5 的读取、校验和视频回放继续支持。

`Simulation.observe()` 返回以下扁平字段，供后续策略适配器使用：

| 仿真观测字段 | 内容 |
|---|---|
| `observation.images.head` | 固定第三视角 RGB，uint8 `[H,W,3]` |
| `observation.images.wrist` | 腕部 RGB，uint8 `[H,W,3]` |
| `observation.depth.head` | 第三视角深度，float32 `[H,W]`，米 |
| `observation.depth.wrist` | 腕部深度，float32 `[H,W]`，米 |
| `observation.state` | 7 个实际关节角与完整夹爪开口，float32 `[8]` |

深度由 `config/insert_plug.yaml` 的 `cameras.depth.enabled` 控制，当前默认开启；关闭后不渲染或保存深度。RGB 和深度在同一物理状态、同一相机、同一分辨率下读取，像素相互对齐。深度是相机光轴方向的 Z 距离，MuJoCo 已返回米制值，不需要再次线性化。数值保持 float32，使用 HDF5 无损压缩；普通 RGB MP4 只供查看 RGB，不承载原始深度。

每条新记录的 `metadata.feature_map` 映射上述图像字段到 HDF5 路径，深度数据集还声明 `units=m`、`convention=camera_optical_axis_z`。`metadata.camera_calibration` 包含内参、近远裁剪距离、记录起始时的光学坐标系外参。腕部外参随运动变化，可用记录的 `simulation/qpos` 和对应 XML 逐帧前向计算；远裁剪面附近的背景深度应在点云重建时屏蔽。

`snapshot` 同时导出 `head.png`、`wrist.png`、`head_depth.npy`、`wrist_depth.npy`、带米制色标的 `depth_preview.png` 和 `calibration.json`。深度预览仅对显示范围做截断，原始 NPY/HDF5 数值不变。这是 RGB-D 采集支持；LeRobot ACT 默认 RGB 骨干仍需额外的深度预处理/编码融合适配才能使用深度，当前没有把深度自动接入训练。

每个完成的采集 episode 保留同名 HDF5、配置 YAML、场景 XML 和第三视角 MP4；结果保存在 HDF5 元数据中，不另存 JSON、PNG 或规划文件。记录时 RGB、深度、机器人状态与动作共享真实仿真时间戳；默认采样和专家控制均为 10 Hz，以 1 kHz 推进接触动力学。`--no-images` 仅执行物理过程，不生成 HDF5 或视频。

当前 10 Hz 采样间隔正好对应 100 个 1 ms 物理步，时间戳为 `0.000、0.100、0.200…` 秒。记录器保留真实仿真时间，元数据记录采样频率、控制频率、物理步长与采样时序约定。历史 `test/` 的 15 Hz 采样使用最近的真实物理步，因此可见 `0.000、0.067、0.133、0.200` 秒等 66/67 ms 间隔，相对理想采样时刻的误差不超过 0.5 ms。

当前 `action[t]` 是该采样时刻已经生效、即将用于下一物理步的关节目标与夹爪开口，并保持到 100 ms 后的下一条观测。专家在下发前将动作量化为 float32，再将这些精确数值转为 float64 供仿真执行；HDF5 保存同一数值，LeRobot 的 float32 导出不会改变实际控制序列。元数据 `action_timing` 声明 `held_until_next_observation`，`action_hold_duration_s=0.1`；`execution_result.execution_checks` 另记录 `action_command_dtype=float32` 和 `control_hold_seconds=0.1`。

旧的 50 Hz 控制／10 Hz 或 15 Hz 记录中，保存动作可能在下一条观测前被专家更新。读取历史记录或显式配置不同的控制、记录频率时，必须依据各条元数据解释动作时序。

| 字段 | 内容 |
|---|---|
| `observations/images/head`、`wrist` | uint8 `[T,H,W,3]` RGB |
| `observations/depth/head`、`wrist` | 可选 float32 `[T,H,W]` 光轴深度，米 |
| `observations/joint_position`、`joint_velocity` | `[T,7]` 实际关节状态 |
| `observations/gripper_width` | `[T,1]` 完整开口，m |
| `action` | `[T,8]`：7 个目标关节角＋完整夹爪开口 |
| `timestamp`、`phase` | 仿真时间、专家阶段 |
| `simulation/qpos`、`qvel`、`ctrl` | 完整仿真状态；ctrl 是采样时生效的执行器控制量 |
| `privileged/*_pose` | 零件与孔的真值位姿，仅供专家和诊断 |

写入中使用 `.h5.partial`，完成后才发布 `.h5`。完整名义预检失败时尚未创建采集文件；执行中被规划或安全检查中止的轨迹保留并标记失败原因。训练前按 `success` 筛选，并显式选择可用观测，避免把物体真值当作视觉策略输入。

以上是本场景的原始 HDF5 格式；完整的十布局批次可通过下述步骤另行转换为 LeRobotDataset。视频回放播放记录帧，不重新计算物理。训练后的策略可使用 `Simulation.observe()` 和 `Simulation.step(action)` 接入。

### LeRobot 转换与 ACT 输入

先前专家的历史批次已合并到 `outputs/datasets/insert_plug/lerobot/`，包含 **100 条、38,410 帧、10 Hz，约 11.65 GB**，保持原样。它此前替换的是更早的 50 Hz 控制批次，不会因为本次专家优化再次删除或更新。该历史转换的全部动作、状态、时间戳、阶段及深度像素逐帧校验通过，动作 float32 转换无损，最终目录也已通过官方读取器验证。转换使用官方 **LeRobot 0.7.0** 的 `LeRobotDataset.create/add_frame/save_episode/finalize` API，写入 [LeRobot v3 格式](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)，仅在本机保存。源 HDF5 保留完整仿真状态和逐帧重建腕部外参所需的信息。

```text
lerobot/
├── data/chunk-000/*.parquet             # 状态、动作、深度像素和来源字段
├── videos/observation.images.head/      # 第三视角 RGB
├── videos/observation.images.wrist/     # 腕部 RGB
└── meta/                               # 官方索引/统计及来源、验证清单
```

转换校验包括官方 reader 读取和双路深度逐像素对照源 HDF5，结果保存在 `meta/export_validation.json`。RGB 使用 H.264，属于有损编码。深度保存为官方支持的 **float32 TIFF image bytes，内嵌 Parquet**，单位米、光轴 Z 距离，没有量化成普通视频，也不依赖临时 TIFF 文件。

| LeRobot 字段 | 内容 |
|---|---|
| `observation.images.head`、`observation.images.wrist` | 两路 RGB；官方 reader 返回 float32 `[3,256,256]`，范围 `[0,1]` |
| `observation.depth.head`、`observation.depth.wrist` | 两路无损深度；返回 float32 `[1,256,256]`，米 |
| `observation.state` | `[8]`：7 个实际关节角 rad ＋实际夹爪开口 m |
| `action` | `[8]`：7 个目标关节角 rad ＋目标夹爪开口 m |
| `source.position_id`、`source.episode_index` | 原始位置编号与全局 episode 编号 |
| `source.timestamp`、`source.phase` | 原始仿真时间与专家阶段 |

LeRobot 自带 `timestamp` 是本 episode 的 `frame_index / fps`；原时间另存于 float64 的 `source.timestamp` 列。官方通用 reader 读取此列时会转为 float32，需双精度时从 Parquet 或原 HDF5 读取。`meta/export_manifest.json` 逐条保存源文件相对路径、位置、帧范围、初始相机标定、深度校验值和动作时序约定；原 HDF5 路径以清单内 `source_root` 为基准。数据集自身的媒体路径是相对路径，可以整体移动。

```bash
conda activate lerobot
cd ~/Projects/mujoco
# 先完成 under20_v1 的十布局采集；目标转换目录须尚不存在
python scripts/export_insert_plug_lerobot.py \
  --source-root outputs/datasets/insert_plug/under20_v1 \
  --output outputs/datasets/insert_plug/under20_v1/lerobot

# 仅在其他机器重建转换环境时执行
conda env create -f environment-lerobot.yml
conda activate lerobot
python -m pip install -r requirements-lerobot-lock.txt
```

转换逐帧读取原始 HDF5，先写 `.partial` 目录，校验后发布；已有目标目录不会覆盖。默认只扫描指定根目录下的 `pos0` 至 `pos9`，不混入旧 `test/` 数据。下面读取的是保留的历史数据：

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/insert_plug",
    root=Path("outputs/datasets/insert_plug/lerobot"),
    video_backend="pyav",
    depth_output_unit="m",
)
frame = dataset[0]
print(dataset.num_episodes, dataset.num_frames)  # 100, 38410
print(frame["observation.depth.head"].shape)      # [1, 256, 256]
```

ACT 的 RGB 输入必须显式限定为两路 `observation.images.*`。LeRobot 自动推断输入时也会将 image 类型的深度视为视觉特征，因此训练配置的 `policy.input_features` 和 `policy.output_features` 应采用 `config/act_insert_plug_features.json` 中的字段；该文件是配置片段，不是完整训练配置。官方 `make_policy` 在显式设置 `input_features` 后会保留这个子集；可用下列初始化代码核对模型只使用 head、wrist 和 8 维 state，输出 8 维 action。

```python
import json
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy

fragment = json.loads(Path("config/act_insert_plug_features.json").read_text())
def features(section):
    return {key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
            for key, value in fragment[section].items()}

config = ACTConfig(input_features=features("input_features"),
                   output_features=features("output_features"),
                   device="cpu", pretrained_backbone_weights=None, push_to_hub=False)
policy = make_policy(config, ds_meta=dataset.meta)  # 仅初始化，尚未训练
```

此独立环境安装的是 CPU PyTorch，用于转换和接口验证。深度单独完整保存，但上述标准 ACT 输入配置只使用 RGB 和机器人状态；将来训练 RGB-D 模型需要另外接入深度编码。当前数据的策略动作、专家控制和记录均为 10 Hz，转换保留实际执行的 float32 控制数值。部署时每个策略动作调用一次 `Simulation.step(action)`，在 1 kHz 物理循环中恒定保持 100 ms，再获取下一次观测；部署效果仍需独立仿真验证。

### 每条轨迹的长度与空间占用

保留的先前专家十布局数据每条执行 37.7–39.4 秒，控制与记录均为 10 Hz，每条 377–394 帧，100 条合计 38,410 帧。双路 RGB-D 均为 256×256；原始四文件记录合计 13,640,790,169 字节（约 13.64 GB），对应 LeRobot 数据约 11.65 GB。原始每条平均约 136.4 MB。当前 20 秒以内的专家十布局验证时长为 18.6–19.7 秒，对应默认 10 Hz 下每条 186–197 帧；新数据实际大小仍取决于画面与压缩效果，不能沿用历史总大小。

## ACT 策略部署

```bash
conda activate mujoco_policy
cd ~/Projects/mujoco
python -m examples.insert_plug rollout --viewer --realtime
```

`config/insert_plug_policy.yaml` 默认指向 `outputs/train/act_insert_plug_20261107_2044/artifacts/checkpoints/030000`，使用 pos0、10 Hz、每次执行 50 个预测动作、最长 60 秒。更换 checkpoint 修改此 YAML 或传 `--checkpoint`，接受训练步目录及 `pretrained_model/` 两种形式。YAML 中相对路径以工作区根目录为基准，命令行 `--output` 相对当前目录。

```bash
# 服务器 / 无桌面运行；默认输出 head | wrist 双视角 MP4
python -m examples.insert_plug rollout --backend egl --position 0
# 每个已采集位置评估一次
python -m examples.insert_plug rollout --backend egl --position all --episodes 1
# 使用 config/insert_plug.yaml 中 source_xy/target_xy
python -m examples.insert_plug rollout --backend egl --position config
# 测试动作序列执行长度，不改变训练的 chunk_size
python -m examples.insert_plug rollout --backend egl --n-action-steps 50
```

`--device auto` 优先使用可用的 CUDA，否则 CPU；可显式指定 `cpu` 或 `cuda`。`--viewer` 需要桌面 DISPLAY，并使用 GLFW。`--realtime` 按墙钟时间节流，推理和渲染较慢时会慢于实时，但每个动作仍对应准确 100 ms 仿真时间。`--no-video` 不保存视频；`--duration` 指最大仿真秒数，`--episodes` 指每个布局的重复次数，`--seed` 控制推理随机种子，不会随机改变布局。

复用接口：

```python
from mujoco_tools.lerobot_policy import LeRobotPolicy

# sim 是已按任务配置创建的 Simulation
policy = LeRobotPolicy('/path/to/checkpoint', device='auto', n_action_steps=50)
sim.reset()
policy.reset()  # 每次新 episode 必须同时清空 ACT 动作队列和处理器状态
observation = sim.observe()
action = policy.select_action(observation)
sim.step(action)  # 一次 100 ms；持续循环直到成功、超时或异常
```

通用适配器在 `src/mujoco_tools/lerobot_policy.py`，不依赖具体 MuJoCo 场景，目前支持 ACT。任务入口 `examples/insert_plug/rollout.py` 负责场景、10 Hz 时序、动作语义检查、成功判定与视频。加载时严格检查模型参数与输入维度，使用 checkpoint 自带的处理器及统计量。HWC uint8 RGB 只做一次 `/255` 和 CHW 转换，再由保存的处理器归一化；状态是实际 7 关节角与完整夹爪开度。动作反归一化后是绝对目标，不能作为位移累加。该模型不输入深度，也不输入物体位置、任务阶段或规划器结果。完整模型已包含视觉骨干权重，加载无需再下载 ImageNet 权重。

当前 checkpoint 的 `chunk_size=100` 保留；本任务 YAML 默认覆盖 `n_action_steps=50`，依次执行 50 个不同动作，5 秒后重新预测；每个动作仍只保持 100 ms。设为 1 可每 100 ms 根据最新观测预测。通用适配器的构造器缺省值仍为 1，任务 YAML 显式选择 50；选择执行长度应对比物理成功率。

输出默认在 `outputs/insert_plug/rollouts/run_<时间>/`，与训练数据分开，已有目录拒绝覆盖。每条轨迹保留视频、实际场景/配置和执行报告，总报告记录模型路径、版本、动作时序、推理耗时、停止原因及接触诊断模式。策略控制完全通过执行器与物理积分，运行中不使用专家补救、IK 修正或位姿瞬移。默认 `contact_diagnostics=false`、`stop_on_contact=false`，不创建 `ReplayContactMonitor`、不因普通接触中止。成功需插入到位、松爪、末端撤离且物体稳定连续至少 0.25 秒，每个物理步仍监测任务完成状态；全部任务成功返回 0，否则返回 2。

需要额外观察接触时使用 `--contact-diagnostics`，此模式只记录旧阈值诊断，正常完成任务仍可记为成功。`--stop-on-contact` 才启用旧严格接触中止；`--no-contact-diagnostics`、`--no-stop-on-contact` 可关闭 YAML 中的相应选项，旧 `--continue-on-contact` 兼容为禁用接触中止。默认报告 `contact_policy="off"`、`contacts.enabled=false`、`contact_checks=0`、`contact_clean=null`，含义是未做附加诊断，不是“零碰撞”。`physics_checks` 记录任务成功监测次数。非法动作、非有限值、关节越界和数值异常仍会报告失败。

推理环境单独使用 `mujoco_policy`，包含兼容的 Python、PyTorch、当前 fork 的 LeRobot 和 MuJoCo；环境定义与锁文件在工作区根目录，重建入口为 `scripts/setup_policy_env.sh`。原 `mujoco` 采集环境与 `lerobot` 数据转换环境保留。该模型的实际任务成功率应读取部署报告；数据集物理动作回放成功并不代表训练策略也会成功。

此前 `030000` 在 pos0 的单条部署验证结果（不代表当前 50 步设置）：

| 执行长度 | 设备 | 结果 |
|---|---|---|
| 1 | CPU | 60 秒超时，停留在初始姿态附近，零违规接触 |
| 100（checkpoint 原值） | CPU | 9.707 秒出现插销与源孔壁违规接触，停止 |
| 10（历史设置） | CUDA | 接近并夹住零件，未继续拔出；60 秒超时，零违规接触 |

结果目录分别为 `outputs/insert_plug/rollouts/act_030000_pos0_n1_cpu/`、`act_030000_pos0_n100_cpu/`、`act_030000_pos0_n10_cuda/`。GPU 运行检查了完整 60,000 个物理步，双视角视频为 512×256、10 fps；60 秒仿真在未开启实时节流时约耗时 51 秒。最早 CPU 烟测的 XML 是根据报告保存的完整配置补建，并在报告中注明；后续运行在启动时直接保存实际 XML。三个设置均未完成任务，这是已测 pos0 的结果，不能推广成全部十个位置的成功率。

完整无窗口批量测试按训练运行名称保存。`outputs/test/act_insert_plug_20261107_2044/` 保留旧严格接触模式的 50 步批次，每个 pos 测试 10 次，共 100 次；更早的 10 步执行批次及其额外诊断已按用户要求删除。保留批次的历史结果为 **3 次完整成功、97 次接触阈值中止、0 次超时**，成功均来自 pos2（3/10）。原 `summary.json`、`episodes.csv` 和视频不改写；97 次中止只能说明触发了当时阈值，不能证明允许继续执行后仍会失败，也不能代表当前默认模式的成功率。

关闭接触中止后另测 pos3、seed 30 一次，完整执行 600 个动作、60 秒，无接触中止或运行异常，但超时未完成任务，见[单次新模式报告](../../outputs/test/act_insert_plug_20261107_2044_contact_response/pos3/summary.json)与[双视角视频](../../outputs/test/act_insert_plug_20261107_2044_contact_response/pos3/episode_000000/rollout.mp4)。视频 601 帧完整解码。这是单次检查，不是 100 条重新评估结果。

后续新模式批量测试使用新目录，例如下列命令；示例不表示已重新执行 100 次测试：

```bash
python scripts/test_insert_plug_policy.py --workers 5 --n-action-steps 50 \
  --output outputs/test/act_insert_plug_20261107_2044_contact_response_batch
python scripts/summarize_insert_plug_policy_test.py \
  --root outputs/test/act_insert_plug_20261107_2044_contact_response_batch \
  --checkpoint outputs/train/act_insert_plug_20261107_2044/artifacts/checkpoints/030000 \
  --n-action-steps 50 --write
```

历史 pos0 逐步诊断发现，完整预测里超过 5 mm 的抬升在第 23–28 个动作才出现，前 10 步主要保持抓取，因此反复重新预测会推迟抬升。这是调整到 50 步的依据。当前每步仍为 100 ms；之后按用户要求关闭了默认附加接触中止，模型完成率需在新模式下另行评估。

## STL 与碰撞几何

```bash
python scripts/inspect_stl.py /path/to/part.stl --units mm
python scripts/prepare_insert_plug_objects.py
```

两个原始 STL 均保留不变。螺栓的杆与六角头分开构建凸碰撞体；放置桶按角度和高度分解为多个凸碰撞体，保留孔腔、内底与桶口。完整 STL 用于外观，避免把带孔桶直接当成一个会封住孔腔的凸包。派生资产目录记录生成方法与几何近似；修改 STL 后需要重新生成碰撞资产并核对配置尺寸。
