# insert_plug 插销任务

FR3 配合 Robotiq 2F-85，从一个放置桶夹起六角头螺栓，沿孔轴拔出、搬运、对准并浅插另一个放置桶，再松爪让零件靠重力落座。

机械臂和夹爪来自固定版本的 MuJoCo Menagerie，保留各自许可证和校验清单。物件外观使用用户提供的 `assets/放置桶.STL` 与 `assets/螺栓.STL`。机器人使用理想重力补偿，零件自由运动并保留重力、摩擦和接触；质量和摩擦系数仍是可调整的仿真假设。

## 文件位置

任务根目录保留统一入口 `cli.py`、`__main__.py`、说明和生成的 `scene.xml`，实现按功能分组：

```text
examples/insert_plug/
├── environment/  # 场景构建、物件准备、仿真接口、布局与截图
├── control/      # 专家控制、轨迹规划与执行检查
├── data/         # 数据格式、记录、批量采集、转换与视频回放
├── policy/       # 模型部署、批量评估、汇总与抬升诊断
└── diagnostics/  # 专家验证、记录审计与动作物理回放检查
```

日常仿真仍使用 `python -m examples.insert_plug`；批量工具使用下文的模块入口。共享配置、资产和输出继续放在工作区顶层。

训练目录名使用 `<model>_insert_plug_<version>_<rgb|rgbd>_<训练时间>`；`outputs/test/` 下的批测目录再加 `ind` / `ood`，末尾使用该次测试的启动时间（`Asia/Shanghai`，`YYYYmmdd_HHMMSS`）。不在目录名中重复 batch size、workers、checkpoint 步数或执行长度。`policy.evaluate` 省略 `--output` 时自动命名；未知布局需显式 `--split ind/ood` 或 `--output`。历史日志和 checkpoint 内保存的原配置保持不变，当前路径与历史校验值的对应关系见[迁移清单](../../outputs/_checks/output_naming_20261112/manifest.json)。

注意力、相机消融、平移诊断与报告分别保存在 `outputs/insert_plug/` 的四个类别目录。实验名包含模型、任务版本、模态、适用的 ind/ood、必要实验类型和时间；具体 seed、checkpoint 步数、执行长度等仍存于元数据。ACT 注意力用 `chunked` / `ensemble`，相机消融用 `ensemble` / `stateless`，旧预测的重新统计另标 `stateless_reanalysis`。新默认名称与报告使用同一次上海时间 `started_at`；未知坐标不标 ind/ood，无法确认版本或模态时显式指定 `--output`。旧实验日期来源及完整映射见[实验迁移清单](../../outputs/_checks/insert_plug_naming_20261112/manifest.json)。

| 内容 | 工作区内路径 |
|---|---|
| 场景与物理参数 | `config/insert_plug.yaml` |
| 10 组已验证的 2 cm 网格位置 | `config/insert_plug_positions.yaml` |
| FR3 模型与许可证 | `assets/robots/franka_fr3/` |
| Robotiq 模型与许可证 | `assets/grippers/robotiq_2f85/` |
| 原始零件网格 | `assets/放置桶.STL`、`assets/螺栓.STL` |
| 从 STL 派生的碰撞几何 | `assets/objects/insert_plug/` |
| 场景构建 | `examples/insert_plug/environment/scene.py` |
| 当前配置生成的完整场景 | `examples/insert_plug/scene.xml` |
| 本任务物件装配 | `examples/insert_plug/environment/objects.py` |
| 通用 Robotiq 装配与开合控制 | `src/mujoco_tools/robotiq.py` |
| 轨迹规划、专家与执行检查 | `examples/insert_plug/control/` |
| 仿真接口 | `examples/insert_plug/environment/simulation.py` |
| 本场景数据格式与视频回放 | `examples/insert_plug/data/dataset.py`、`examples/insert_plug/data/replay.py` |
| 本批次采集数据（每条 HDF5、YAML、XML、第三视角 MP4） | `outputs/datasets/insert_plug_v1/pos0`～`pos9` |
| 本批次合并后的 LeRobot v3 数据集 | `outputs/datasets/insert_plug_v1/lerobot/` |
| LeRobot 转换入口与 ACT 输入配置 | `examples/insert_plug/data/export_lerobot.py`、`config/act_insert_plug_features.json` |
| 按需生成的图片、视频、计划导出 | `outputs/insert_plug/` |

需要专项检查时，可用以下入口的 `--help` 查看参数；检查结果放在 `outputs/_checks/`：

| 用途 | 查看参数 |
|---|---|
| 专家执行与布局验证 | `python -m examples.insert_plug.diagnostics.validate_expert --help` |
| 轨迹运动密度审计 | `python -m examples.insert_plug.diagnostics.audit_motion_density --help` |
| 策略抬升诊断 | `python -m examples.insert_plug.policy.diagnose_lift --help` |

## 运行

以下命令均在工作区顶层执行：

```bash
conda activate mujoco
cd ~/Projects/MuJoCo

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
python -m examples.insert_plug validate outputs/datasets/insert_plug_v1/pos0/episode_000000/episode_000000.h5
python -m examples.insert_plug preview outputs/datasets/insert_plug_v1/pos0/episode_000000/episode_000000.h5 --output outputs/insert_plug/replay.mp4
```

每条记录默认生成同名 `.mp4`，只包含第三视角 `head` RGB，以 HDF5 的采样频率播放（默认 10 fps），不叠字、不缩放，帧数与 HDF5 一致。带 `--video` 的上述命令还会额外将双视角视频另存到 `outputs/insert_plug/previews/under20_demo/preview.mp4`。额外视频左侧为 Head (third-person)，右侧为 Wrist；独立黑色页眉显示相机名称、记录的仿真时间和当前阶段。已有 episode 和视频不会覆盖，也不会随当前配置自动更新；重复执行这条命令请改用新的输出目录。

省略 `--output` 时，`demo` 与 `collect` 都保存到 `outputs/datasets/insert_plug/run_<时间>/`；指定 `--output` 时使用该批次目录。每个 episode 必须放在独立的 `episode_<seed>/` 子目录中，内部保留同名 `.h5`、`.yaml`、`.xml`、`.mp4` 四个文件。历史 15 Hz 固定位置测试批的原始保存路径为 `outputs/datasets/insert_plug/test/`，结构如下：

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
python -m examples.insert_plug.data.collect_positions --workers 3 \
  --output outputs/datasets/insert_plug_v1

# 可单独选择某组；重复运行会校验并跳过同配置的成功记录
python -m examples.insert_plug.data.collect_positions --position pos0 --workers 1 \
  --output outputs/datasets/insert_plug_v1

# 整批完成后，检查位置、频率、深度、动作时序、视频及每条时长严格小于 20 秒
python -m examples.insert_plug.diagnostics.audit_batch --positions-config config/insert_plug_positions.yaml \
  --root outputs/datasets/insert_plug_v1 --sample-hz 10 --control-hz 10 --max-duration-s 20
```

本次新批次已在 `outputs/datasets/insert_plug_v1/pos0`～`pos9` 各采集 10 条，**100/100 成功，共 18,990 帧、400 个原始文件，6,549,365,746 字节（约 6.55 GB）**。完整任务时长 **18.6–19.7 秒，平均 18.99 秒**，全部通过严格 `<20s` 的[原始数据审计](../../outputs/_checks/insert_plug_under20_collection/raw_audit.json)。本次使用 OSMesa CPU 渲染，记录 256×256 双路 RGB-D；控制和记录均为 10 Hz，物理为 1 kHz。转换后的 `lerobot/` 与各位置目录并列。

先前专家的历史批次位于原始保存路径 `outputs/datasets/insert_plug/`，不随代码克隆迁移；其 100 条记录采用 10 Hz 控制、10 Hz 记录、1 kHz 物理，并曾替换更早的 50 Hz 控制／10 Hz 记录批次。该历史批次 **100/100 条成功，共 38,410 帧、400 个文件，约 13.64 GB**，采集时通过整批独立审计和保存动作物理回放；初始和最终视觉网格的画面边缘最小余量为 51.694 px。这些历史统计与本次新批次分别保存。本次原始记录结构为：

```text
outputs/datasets/insert_plug_v1/
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
├── pos9/
│   └── episode_000090/ ... episode_000099/
└── lerobot/                   # 本批次转换后的 LeRobot v3 数据集
```

`pos{i}` 使用全局编号 `i*10` 至 `i*10+9`。本批次直接使用 `outputs/datasets/insert_plug_v1/`，不添加中间目录。基础配置和批量清单的控制、记录频率均为 10 Hz，批量采集开启两路深度；原 `test/` 中的历史 15 Hz 记录保持原样。每条 HDF5 和 YAML 都保存本次实际配置与所属位置；每条动作在采样时生效，并保持至 100 ms 后的下一次观测。

无 GPU/EGL 渲染时，可在采集命令中添加 `--backend osmesa`，使用 CPU 无窗口渲染；数据格式和仿真频率相同。

安全续采仅接受四个文件齐全、HDF5 校验成功且 YAML 配置完全一致的已有记录。遇到缺失文件、失败轨迹或配置不同会停止并保留现有内容。基础配置 SHA256 若与清单不一致，也会拒绝启动，避免使用过时的位置预检结果。

需要根据修改后的基础配置生成新布局时，使用新的清单和输出目录：

```bash
python -m examples.insert_plug.environment.layouts --config config/insert_plug.yaml --output config/insert_plug_positions_new.yaml
python -m examples.insert_plug.data.collect_positions --positions-config config/insert_plug_positions_new.yaml --output outputs/datasets/insert_plug/new_positions --workers 3
```

布局生成器默认不覆盖已有清单；`--position` 可重复指定以选择多个位置。后续新批次完成后，可通过下述独立转换步骤合并为 LeRobotDataset，原始四类文件仍保留。

## 10 Hz 动作物理回放检查

`examples/insert_plug/diagnostics/check_action_replay.py` 默认逐条独立执行 `pos0` 至 `pos9` 的全部 100 条保存动作，只读原始记录，报告默认写入 `outputs/_checks/insert_plug_action_replay_<时间>/report.json`。在工作区顶层运行：

```bash
conda activate mujoco
env -u PYTHONPATH python -m examples.insert_plug.diagnostics.check_action_replay \
  --root outputs/datasets/insert_plug_v1

# 可用 --episode 选择单条，重复该参数可选择多条
env -u PYTHONPATH python -m examples.insert_plug.diagnostics.check_action_replay \
  --episode outputs/datasets/insert_plug_v1/pos0/episode_000000/episode_000000.h5 \
  --workers 1
```

`--action-dtype` 默认为 `float32`，与 LeRobot 导出动作一致；指定 `float64` 可检查原始 HDF5 精度。当前新记录在实际下发前已量化为 float32，因此这两种读取方式的控制数值相同。每条从配套场景的正常 reset 开始，读取记录内的控制频率：当前 10 Hz 控制／10 Hz 记录的每个动作只执行一次 100 ms 的 `Simulation.step()`；旧的 50 Hz 控制／10 Hz 记录则将每个目标重复执行五次 20 ms 控制步，并将末动作截断到原执行时长。两种情况下均由执行器和 1 kHz 物理推进，不恢复中间状态、不调用专家。这是保存动作的独立物理验证，不属于策略评估或录像回放。

报告中的 `task_success` 和 `success` 要求完整执行、最终落座条件通过，并在最后 250 ms 的 251 个物理时刻全部稳定；稳定性指标在独立 `MjData` 副本上刷新计算。默认关闭附加接触诊断，`contact_clean=null`、`contacts.enabled=false`、接触检查次数为 0，表示未检测。显式传入 `--contact-diagnostics` 可记录旧阈值诊断；它不依赖原阶段标签、不中止回放，也不否决物理成功。

本次 `insert_plug_v1` 的 [100 条独立动作回放](../../outputs/_checks/insert_plug_under20_collection/action_replay.json)全部物理成功，与源记录的关节及插销状态最大偏差均为零；每条最终 250 ms 的 251 个物理采样点均稳定。接触诊断保持关闭，因此 `contact_clean=null`，不能解释为零接触或零违规。该结果验证保存动作能复现专家轨迹，不是训练策略成功率。

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

上一版连续路点专家平均 24.42 秒，报告与视频保留在 [insert_plug_continuous_hover](../../outputs/_checks/insert_plug_continuous_hover/summary.json) 和 [continuous_hover.mp4](../../outputs/insert_plug/previews/continuous_hover.mp4)。再前一版平滑专家平均 30.3 秒，保留训练批次平均 38.41 秒。旧平滑版的[严格模式报告](../../outputs/_checks/insert_plug_smooth_expert/summary.json)、[关闭附加接触中止的报告](../../outputs/_checks/insert_plug_smooth_expert/contact_response/summary.json)及 [smooth_expert.mp4](../../outputs/insert_plug/previews/smooth_expert.mp4)均为历史对照；旧版 313 帧 RGB-D 试录、精确动作回放和 75 项回归测试也属于当时的验证。该次专家优化未改写历史数据、LeRobot 数据、checkpoint 或策略测试；后续新批次训练状态见工作区说明，模型成功率仍需独立评估。抓稳、对准、插入和最终稳定要求均保留；默认接触诊断仍关闭，相关字段为 disabled/null，不声明“零违规接触”。

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

本次转换已完成，目录为 **`outputs/datasets/insert_plug_v1/lerobot/`**，使用共享 `mujoco` 环境内的本地 **LeRobot 0.7.1** 写入 v3 格式：**100 条、18,990 帧、10 Hz，108 个文件，5,708,651,034 字节（约 5.71 GB）**。全部 18,990 帧的双路深度逐位一致，动作与状态匹配源 float32 数值，原始时间戳和阶段完全一致；动作转换无损。官方 reader 检查 300 个样本，发布目录另检查 12 个样本，均通过。RGB 使用有损 H.264，本次抽检最坏平均绝对误差为 1.319（0–255）。见[转换验证](../../outputs/datasets/insert_plug_v1/lerobot/meta/export_validation.json)与[本批次总报告](../../outputs/_checks/insert_plug_under20_collection/summary.json)。原始记录保持完整；后续 ACT 与 Diffusion 均已完成 30,000 步，详见[工作区训练状态](../../README.md#训练-diffusion)。

先前专家历史转换的原始保存路径为 `outputs/datasets/insert_plug/lerobot/`，包含 **100 条、38,410 帧、10 Hz，约 11.65 GB**，保持原样。它此前替换的是更早的 50 Hz 控制批次，不会因为本次专家优化再次删除或更新。该历史转换的全部动作、状态、时间戳、阶段及深度像素逐帧校验通过，动作 float32 转换无损，最终目录也已通过官方读取器验证。该历史转换使用官方 **LeRobot 0.7.0** 的 `LeRobotDataset.create/add_frame/save_episode/finalize` API，写入 [LeRobot v3 格式](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)，仅在本机保存。源 HDF5 保留完整仿真状态和逐帧重建腕部外参所需的信息。

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
conda activate mujoco
cd ~/Projects/MuJoCo
# 本批次已完成转换；以下为复现命令，重做时须将 --output 改为尚不存在的新目录
python -m examples.insert_plug.data.export_lerobot \
  --source-root outputs/datasets/insert_plug_v1 \
  --output outputs/datasets/insert_plug_v1/lerobot
```

转换逐帧读取原始 HDF5，先写 `.partial` 目录，校验后发布；已有目标目录不会覆盖。默认只扫描指定根目录下的 `pos0` 至 `pos9`，不混入旧 `test/` 数据。下面读取的是本次新批次：

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/insert_plug",
    root=Path("outputs/datasets/insert_plug_v1/lerobot"),
    video_backend="pyav",
    depth_output_unit="m",
)
frame = dataset[0]
print(dataset.num_episodes, dataset.num_frames)  # 100, 18990
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

转换、训练和部署均使用统一的 `mujoco` 环境及本地 `third_party/lerobot[training,diffusion]`；上述初始化示例显式选择 CPU，也可按需使用 CUDA。训练入口与完整 ACT 命令见[工作区训练说明](../../README.md#训练-act)。深度单独完整保存，但上述标准 ACT 输入配置只使用 RGB 和机器人状态；将来训练 RGB-D 模型需要另外接入深度编码。当前数据的策略动作、专家控制和记录均为 10 Hz，转换保留实际执行的 float32 控制数值。部署时每个策略动作调用一次 `Simulation.step(action)`，在 1 kHz 物理循环中恒定保持 100 ms，再获取下一次观测；部署效果仍需独立仿真验证。

### 每条轨迹的长度与空间占用

本次 `insert_plug_v1` 每条执行 18.6–19.7 秒，控制与记录均为 10 Hz，每条 186–197 帧，100 条合计 18,990 帧。双路 RGB-D 均为 256×256；400 个原始文件合计 6,549,365,746 字节（约 6.55 GB），每条平均约 65.5 MB；转换后 LeRobot 数据另占 5,708,651,034 字节（约 5.71 GB）。

先前专家的历史批次每条 37.7–39.4 秒、377–394 帧，合计 38,410 帧；原始四文件记录合计 13,640,790,169 字节（约 13.64 GB），对应 LeRobot 数据约 11.65 GB。这些历史大小不适用于本次新批次。

## ACT 策略部署

```bash
conda activate mujoco
cd ~/Projects/MuJoCo
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

通用适配器在 `src/mujoco_tools/lerobot_policy.py`，不依赖具体 MuJoCo 场景，支持 ACT 和 Diffusion，按 checkpoint `type` 自动选择。任务入口 `examples/insert_plug/policy/rollout.py` 负责场景、10 Hz 时序、动作语义检查、成功判定与视频。加载时严格检查模型参数与输入维度，使用 checkpoint 自带的处理器及统计量。HWC uint8 RGB 只做一次 `/255` 和 CHW 转换，再由保存的处理器归一化；状态是实际 7 关节角与完整夹爪开度。动作反归一化后是绝对目标，不能作为位移累加。这两份 checkpoint 不输入深度、物体位置、任务阶段或规划器结果。完整模型已包含视觉骨干权重，加载无需再下载 ImageNet 权重。

上述旧 checkpoint 的 `chunk_size=100` 保留；本任务 YAML 默认覆盖 `n_action_steps=50`，依次执行 50 个不同动作，5 秒后重新预测；每个动作仍只保持 100 ms。新 `act_insert_plug_v1_rgb_20261108_154950` checkpoint 保存的是 `chunk_size=64`、`n_action_steps=64`，不要混用旧配置。设为 1 可每 100 ms 根据最新观测预测。通用适配器的构造器缺省值仍为 1，任务 YAML 显式选择 50；选择执行长度应对比物理成功率。

输出默认在 `outputs/insert_plug/rollouts/run_<时间>/`，与训练数据分开，已有目录拒绝覆盖。每条轨迹保留视频、实际场景/配置和执行报告，总报告记录模型路径、版本、动作时序、推理耗时、停止原因及接触诊断模式。策略控制完全通过执行器与物理积分，运行中不使用专家补救、IK 修正或位姿瞬移。默认 `contact_diagnostics=false`、`stop_on_contact=false`，不创建 `ReplayContactMonitor`、不因普通接触中止。成功需插入到位、松爪、末端撤离且物体稳定连续至少 0.25 秒，每个物理步仍监测任务完成状态；全部任务成功返回 0，否则返回 2。

需要额外观察接触时使用 `--contact-diagnostics`，此模式只记录旧阈值诊断，正常完成任务仍可记为成功。`--stop-on-contact` 才启用旧严格接触中止；`--no-contact-diagnostics`、`--no-stop-on-contact` 可关闭 YAML 中的相应选项，旧 `--continue-on-contact` 兼容为禁用接触中止。默认报告 `contact_policy="off"`、`contacts.enabled=false`、`contact_checks=0`、`contact_clean=null`，含义是未做附加诊断，不是“零碰撞”。`physics_checks` 记录任务成功监测次数。非法动作、非有限值、关节越界和数值异常仍会报告失败。

推理与仿真采集、数据转换和训练统一使用 `mujoco`，包含 Python 3.12、PyTorch 2.9.1+cu128、当前 fork 的 LeRobot 和 MuJoCo。环境定义 `environment.yml` 与完整 pip 锁文件 `requirements-lock.txt` 位于工作区根目录，首次配置或更新运行 `bash scripts/setup_mujoco_env.sh`。该模型的实际任务成功率应读取部署报告；数据集物理动作回放成功并不代表训练策略也会成功。

### Diffusion：每次执行 32 步

当前 Diffusion 已训练完 30,000 步。以 32 步执行长度运行一个布局：

```bash
python -m examples.insert_plug rollout \
  --checkpoint outputs/train/diffusion_insert_plug_v1_rgb_20261108_220529/artifacts/checkpoints/030000 \
  --position 0 --n-action-steps 32 --device cuda --backend egl --duration 60
```

训练的 `horizon=64`、`n_obs_steps=1` 保持不变；一次生成预测后，依次执行 32 个 100 ms 动作，3.2 秒后重新预测。保留 DDPM 100 步去噪、RGB MEAN_STD 和状态/动作 MIN_MAX 处理器，不使用 ACT 时序集成。`n_action_steps` 必须不超过 `horizon - n_obs_steps + 1`；改变执行长度不会修改保存的 checkpoint。不同 seed 控制 Diffusion 采样噪声，固定布局不随机变化。

十位置各十次的完整测试仍使用现有批量入口，结果按 pos0–pos9 保存：

```bash
python -m examples.insert_plug.policy.evaluate \
  --checkpoint outputs/train/diffusion_insert_plug_v1_rgb_20261108_220529/artifacts/checkpoints/030000 \
  --output outputs/test/diffusion_insert_plug_v1_rgb_ind_20261109_001724 \
  --n-action-steps 32 --device cuda --backend egl --workers 10 --duration 60
python -m examples.insert_plug.policy.summarize \
  --root outputs/test/diffusion_insert_plug_v1_rgb_ind_20261109_001724 \
  --checkpoint outputs/train/diffusion_insert_plug_v1_rgb_20261108_220529/artifacts/checkpoints/030000 \
  --n-action-steps 32 --duration 60 --video-check full --write
```

已有测试目录拒绝覆盖；重复评估需换新目录。并发数按显存调整，本机 RTX 4090 的十路推理已检查显存容量。批量配置和逐条结果会记录实际模型类型、训练/执行长度、去噪设置及归一化方式，独立审计会检查这些字段并完整解码视频。成功判定与 ACT 一致：实际插入、释放、撤离后持续稳定至少 250 ms；普通接触保留，默认不做附加碰撞阈值中止。

本次最终 `030000`、执行长度 32 的测试已完成：**89/100 成功，11 次 60 秒超时，0 运行异常**。每个固定位置十次，seed 0–99；这是固定场景重复测试，不代表随机布局泛化。

| 位置 | 成功 / 次数 |
|---|---:|
| pos0 | 8/10 |
| pos1 | 4/10 |
| pos2 | 10/10 |
| pos3 | 10/10 |
| pos4 | 9/10 |
| pos5 | 10/10 |
| pos6 | 10/10 |
| pos7 | 10/10 |
| pos8 | 8/10 |
| pos9 | 10/10 |

成功轨迹平均 19.30 秒，范围 17.8–50.5 秒；整批累计 2,378.1 秒仿真。100 条视频共 23,881 帧全部完整解码，所有报告及实际模型配置审计通过，7 个 checkpoint 文件和 3 个源配置哈希保持不变。相关 183 项测试通过。[完整汇总](../../outputs/test/diffusion_insert_plug_v1_rgb_ind_20261109_001724/summary.json)、[逐条 CSV](../../outputs/test/diffusion_insert_plug_v1_rgb_ind_20261109_001724/episodes.csv)和[成功视频示例](../../outputs/test/diffusion_insert_plug_v1_rgb_ind_20261109_001724/pos0/episode_000000/rollout.mp4)保存在测试目录。保留全部失败视频；关闭接触诊断不表示零接触。

### 未见位置对的 ACT / Diffusion 对比

新的 [布局清单](../../config/insert_plug_generalization_v1.yaml) 使用 `heldout_coordinate_range` 划分，包含十个训练中未出现的位置对。选择前已核对 `insert_plug_v1` 的全部 100 份原始 episode YAML，以及 ACT、Diffusion 最终 checkpoint 保存的训练数据配置；原训练的十对位置各对应十条数据。新清单中的 `pos0`–`pos9` 仅是本清单内部编号，与训练清单中同名编号的坐标没有对应关系；每行还带独立的 `heldout_layout_id`。

使用种子 `20261109` 在原 2 cm 网格之间选择新点，源位置、目标位置分别都没有在训练中出现，各自在这十个布局中也互不重复；两类点距相应训练点的最小距离均为 **14.14 mm**，完整四维位置对距训练位置对至少 **20 mm**。所有坐标都在训练数据的逐轴范围内，但这不等于全部位于训练四维位置对凸包内，不能将本组统称为严格插值测试。此次只改变源孔和目标孔的 XY，保持物体、外观、相机、初始机器人状态和其他仿真配置一致，不添加 episode 间位置扰动。

这十对已在策略测试前通过 **10/10 专家真实物理可行性验证**，仿真时长 18.7–19.6 秒；首尾 head 相机的完整可见物体网格投影边距均满足 5 px 要求，实际最小值为 54.19 px。该检查没有测试遮挡，接触诊断保持关闭/null。来源哈希、固定选择计划和逐布局结果及动作轨迹保存在 [验证目录](../../outputs/_checks/insert_plug_generalization_v1/)；专家成功不代表任一模型已经通过测试。

双方使用各自最终 `030000` checkpoint，统一 `n_action_steps=32`，ACT **关闭时序集成**；10 Hz 控制、1 kHz 物理、每条最多 60 秒，保持实际插入、释放、撤离及至少 250 ms 稳定的成功条件。每模型在每个新布局运行十次，均使用 seed 0–99，合计各 100 条、共 200 条。十次重复用于观察同一布局下的表现，空间泛化样本仍只有十个布局。

在共享 `mujoco` 环境中用以下命令运行新的批次，输出必须使用尚不存在的新目录：

```bash
python -m examples.insert_plug.policy.compare_generalization \
  --act-checkpoint outputs/train/act_insert_plug_v1_rgb_20261108_154950/artifacts/checkpoints/030000 \
  --diffusion-checkpoint outputs/train/diffusion_insert_plug_v1_rgb_20261108_220529/artifacts/checkpoints/030000 \
  --positions-config config/insert_plug_generalization_v1.yaml \
  --output "outputs/test/act_diffusion_insert_plug_v1_rgb_ood_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)" \
  --n-action-steps 32 --duration 60 --workers 4 --device cuda --backend egl
```

`--workers 4` 是两种模型共同使用的一个进程并发限额，全部作业共用调度池。输出默认分为 `act/` 和 `diffusion/`，各自按 `pos*/episode_*/` 保留每条报告、场景 XML 和双视角视频；也可分别用 `--act-output`、`--diffusion-output` 指定新的批次目录。入口结束前会审计全部逐条报告、实际配置及来源/checkpoint 哈希，并完整解码双方全部视频，然后生成各模型的 `summary.json`、`episodes.csv`，以及对比目录中的 `summary.json`、`positions.csv` 和 `comparison_config.json`。以 `comparison_ready=true` 表示批次完整且审计通过，实际任务成功率另外统计；超时和任务失败视频也全部保留。

本次对比已完成，使用共享 14 个 worker、CUDA/EGL；新 `pos0`–`pos9` 在两种模型下均为 **0/10 成功**：

| 模型 | 成功 | 60 秒超时 | 运行异常 |
| --- | ---: | ---: | ---: |
| ACT，执行 32 步、关闭时序集成 | 0/100 | 100 | 0 |
| Diffusion，执行 32 步、DDPM 100 | 0/100 | 100 | 0 |

双方各完成 60,000 个控制区间、6,000 秒仿真，200 条视频共 **120,200 帧**全部完整解码。`comparison_ready=true`，独立逐条配置/seed 审计通过；108 项训练/冻结证据引用 SHA 与另外 19 个受保护的模型及配置文件均未改变。新增布局和评估入口共 83 项相关测试通过。运行保留在原训练 tmux 会话的 `generalization_n32` 窗口，进程退出码 0；启动脚本和日志在 [检查目录](../../outputs/_checks/generalization_v1_n32_evaluation/)。

结果在 [对比目录](../../outputs/test/act_diffusion_insert_plug_v1_rgb_ood_20261109_004335/)：包含 [完整汇总](../../outputs/test/act_diffusion_insert_plug_v1_rgb_ood_20261109_004335/summary.json)、[逐位置 CSV](../../outputs/test/act_diffusion_insert_plug_v1_rgb_ood_20261109_004335/positions.csv)、[成功率图](../../outputs/test/act_diffusion_insert_plug_v1_rgb_ood_20261109_004335/comparison.png)、[训练与新位置图](../../outputs/test/act_diffusion_insert_plug_v1_rgb_ood_20261109_004335/positions.png)和 [终点诊断](../../outputs/test/act_diffusion_insert_plug_v1_rgb_ood_20261109_004335/endpoint_diagnostics.json)。全部报告和原视频分别保存在 [ACT 批次](../../outputs/test/act_insert_plug_v1_rgb_ood_20261109_004335/) 与 [Diffusion 批次](../../outputs/test/diffusion_insert_plug_v1_rgb_ood_20261109_004334/)，按 `pos*/episode_*/` 组织。

终点位置统计中，ACT 70 条、Diffusion 83 条的插销尖端 XY 距源位置不超过 1 mm；其余分别为 30 条和 17 条。这个统计只描述终点，不能等同于全过程是否提起。抽查视频明确看到两类代表现象：[ACT pos0](../../outputs/test/act_insert_plug_v1_rgb_ood_20261109_004335/pos0/episode_000000/rollout.mp4) 未把插销从源孔带走；[ACT pos3](../../outputs/test/act_insert_plug_v1_rgb_ood_20261109_004335/pos3/episode_000000/rollout.mp4) 与 [Diffusion pos0](../../outputs/test/diffusion_insert_plug_v1_rgb_ood_20261109_004334/pos0/episode_000000/rollout.mp4) 提起后将其放在目标孔旁，前者倾倒、后者近直立留在桌面。截图与对应时间保存在检查目录的 `previews/`，不能据此断言注意力或内部策略机制。

实验盘点补核了已有的 [ACT 训练位置完整基线](../../outputs/test/act_insert_plug_v1_rgb_ind_20261108_172134/summary.json)：10/100 成功（仅 pos6 为 10/10），90 次超时、0 运行异常，55,930 帧视频完整审计通过；checkpoint 文件哈希、基础场景和 n32／关闭集成等设置与本批一致。因此同设置的训练位置→新位置结果为 **ACT 10/100→0/100、Diffusion 89/100→0/100**，四批正式评测合计 400 条。此前“ACT 缺少完整基线”的表述已更正。这次只评估位置变化，独立新场景数量为十个；不能外推到未见物体、外观、光照或动力学条件。

### Diffusion 在训练内位置上的空间权重可视化

当前 Diffusion checkpoint 使用每相机独立的 ResNet18、32 通道 SpatialSoftmax 和条件 U-Net。可视化提取实际视觉编码时的空间概率图，显示 32 个关键点概率图的等权平均与实际期望坐标。它描述视觉池化分布，不是动作 query 的 Transformer 交叉注意力或因果像素归因；每个相机、每个关键点独立归一化，不能用总概率比较两个视角的重要性。

```bash
python -m examples.insert_plug.policy.visualize_diffusion_attention \
  --checkpoint outputs/train/diffusion_insert_plug_v1_rgb_20261108_220529/artifacts/checkpoints/030000 \
  --position 0 --seed 0 --n-action-steps 32 --duration 60 \
  --device cuda --backend egl
```

默认输出到新的 `outputs/insert_plug/attention/diffusion_insert_plug_<version>_<rgb|rgbd>_[ind|ood]_<时间>/`，已有目录拒绝覆盖。视频上排是实时第三视角和腕部画面，下排是产生当前动作序列时的图像、平均概率图与关键点。保留每次执行 32 步的设置时，每 3.2 秒进行一次新视觉编码；缓存动作期间继续显示原预测图，明确标注源时间及图像年龄。关键点是特征分布的期望坐标，不是带物体语义的检测点。显示固定色标及裁剪格数，原始 8×8 概率不裁剪。

已完成 [InD pos0 / seed0 实例](../../outputs/insert_plug/attention/diffusion_insert_plug_v1_rgb_ind_20261109/)：**19.0 秒成功，190 个动作、6 次视觉编码**，编码时刻为 0、3.2、6.4、9.6、12.8、16.0 秒。[空间权重视频](../../outputs/insert_plug/attention/diffusion_insert_plug_v1_rgb_ind_20261109/attention.mp4)完整解码 190 帧，[普通执行视频](../../outputs/insert_plug/attention/diffusion_insert_plug_v1_rgb_ind_20261109/rollout.mp4)包含最终物理画面，共 191 帧。还有 [六次预测总览](../../outputs/insert_plug/attention/diffusion_insert_plug_v1_rgb_ind_20261109/prediction_overview.png) 和 [9.6 秒预览](../../outputs/insert_plug/attention/diffusion_insert_plug_v1_rgb_ind_20261109/previews/chunk_0003.png)，可用于汇报。

`chunks/` 保存逐次原图、状态、双相机 32×8×8 概率与关键点；`keypoints/` 显示每个通道，色标统一为 0–1；`trace.npz` 保存全部实际观测和动作，`rng_states.npz` 保存随机状态。默认会移除 hook、重置策略并以相同 seed 重放全部固定观测，核对动作逐位一致和 Torch CPU/CUDA 随机状态首尾一致。此实例 190 个动作最大差异为 **0**，结果与原训练位置评测的成功标志、耗时和最终物理指标完全吻合。全部模型/源配置哈希未变，31 项相关测试通过。见 [报告](../../outputs/insert_plug/attention/diffusion_insert_plug_v1_rgb_ind_20261109/report.json)及 [独立归档检查](../../outputs/_checks/diffusion_ind_attention/artifact_audit.json)。该单条可视化另行记录，不增加先前四批正式评测的样本数。

### 动作对图像的注意力可视化

运行已训练 ACT，开启时序集成并逐步生成当前图像的注意力视频：

```bash
conda activate mujoco
python -m examples.insert_plug.policy.visualize_attention \
  --checkpoint outputs/train/act_insert_plug_v1_rgb_20261108_154950/artifacts/checkpoints/030000 \
  --temporal-ensemble-coeff 0.01 --n-action-steps 1 \
  --position 0 --duration 20 --device cpu --backend osmesa
```

默认输出到 `outputs/insert_plug/attention/act_insert_plug_<version>_<rgb|rgbd>_[ind|ood]_<chunked|ensemble>_<时间>/`，包含 `attention.mp4`、`preview.png`、实际场景 `scene.xml` 和 `report.json`。每次预测的完整权重及源图像保存在 `chunks/chunk_*.npz`，抽样预览保存在 `previews/`；开启集成时每秒保存一张 PNG，视频和原始权重仍覆盖每个控制步。已有目录拒绝覆盖。视频上排是当前 head / wrist 画面，下排是在**产生对应预测时的原始图像**上叠加的热力图。标题标出仿真时间、预测编号和 action query。默认 CPU 和两个 PyTorch 线程，无需占用训练 GPU。

`--temporal-ensemble-coeff 0.01` 在推理时开启 LeRobot 原有的时序集成，`n_action_steps` 必须为 1；开启集成且省略执行长度时会自动设为 1，显式传入大于 1 的值会报错。仍然每次预测 64 个未来动作，但每 100 ms 都重新读取当前图像、预测一次，并融合历史预测中对应当前时刻的动作后执行一步，无需重新训练或修改 checkpoint。适配器分别记录 checkpoint 保存的参数和本次实际参数。

集成模式下，热力图显示**本次预测的 query 0**，每步的图像年龄为零；它只是实际执行动作的最新一项贡献。画面底部另外显示所有参与预测的混合权重、最新/历史预测的占比。当前 LeRobot 按从旧到新的顺序使用 `exp(-系数 × 排名)` 再归一化，正数 0.01 会略偏向较早预测，0 表示均匀加权。历史预测的注意力仍保存在各自源图像上，不直接平均并叠到当前画面。报告逐帧保存每项贡献的源时间、query 索引及权重，最多融合 64 次预测。

未传集成系数时保留 checkpoint 的设置；当前 v1 保存的是未开启集成、`n_action_steps=64`，此时每 6.4 秒预测一次，按 query 0–63 显示原始观测上的注意力。只指定 `--n-action-steps 1` 而未启用集成，则每次使用最新预测的首个动作，不混合历史动作。这些选项会改变策略执行行为，不只是显示方式。

共享提取器 `src/mujoco_tools/act_attention.py` 通过只读 forward hook 读取最后一层 decoder cross-attention，保留全部动作 query，并采用 PyTorch 已返回的多头平均值；不修改动作、模型前向参数或权重。当前两路 256×256 图像分别对应 8×8 的骨干特征网格，加上 latent 和 state，共 130 个 memory token。网格尺寸从实际前向输出读取，图像顺序遵循模型配置。原始 NPZ 保存完整 query×token 权重、各相机 query×H×W 权重、源 RGB 和状态；报告保存每帧执行动作、相机/非图像 token 的权重总和及映射关系。

颜色范围固定为每 token 0–0.05，可用 `--color-max` 调整；两相机、全部时刻共用同一范围，不分别归一化。超过上限的颜色饱和，原始权重保持不变，报告记录饱和网格数量。每个相机标注的 token mass 是该相机空间位置的权重之和，剩余权重属于 latent/state 等非图像 token。双线性插值仅用于显示，不代表像素级分辨率。

热力图描述的是**动作 query 对带空间索引的 encoder memory 的关注**。Encoder 自注意力已混合不同位置和状态信息，CNN 特征也有较大的感受野，因此它不是某个关节的独立解释，也不能证明亮区导致了动作；若要研究因果影响，需要另外做遮挡或输入扰动实验。视频生成成功也不代表插入成功，实际停止原因和物理任务结果记录在 `report.json` 的 `rollout` 中；该工具正常生成可视化时返回 0，即使策略超时，运行异常则报错。

已生成当前 v1 / 030000、pos0、seed 1000、系数 0.01、执行长度 1 的[时序集成视频](../../outputs/insert_plug/attention/act_insert_plug_v1_rgb_ind_ensemble_20261108/attention.mp4)及[第 10 秒示例图](../../outputs/insert_plug/attention/act_insert_plug_v1_rgb_ind_ensemble_20261108/previews/chunk_0100.png)。20 秒、10 fps，共 200 个完整动作区间和 200 次模型预测；全部帧均为当前图像、query 0、source age 0，所有历史贡献及原始注意力权重均已核对，视频完整解码 200 帧。此次无运行异常，但 20 秒超时未完成插入。81 项适配器、注意力、时序及 rollout 测试通过；集成后的实际 checkpoint 三步动作序列在有/无采集 hook 时逐位一致，checkpoint 配置未变。运行及检查结果见同目录 `report.json`。

此前执行长度 64、未开启集成的[实际仿真视频](../../outputs/insert_plug/attention/act_insert_plug_v1_rgb_ind_chunked_20261108/attention.mp4)和[缓存动作示例图](../../outputs/insert_plug/attention/act_insert_plug_v1_rgb_ind_chunked_20261108/cached_action_preview.png)保留作为对照：200 个动作、20 秒、10 fps，仅 4 次模型预测，同样超时未完成插入。报告中的 `execution_status` 区分已完整执行、尝试但未完整执行、尚未尝试的动作区间，避免把运行失败前刚选出的动作当成已经执行完成。

### ACT 成功轨迹的图像编码器特征

已生成 [InD pos6 编码器特征视频](../../outputs/insert_plug/attention/act_insert_plug_v1_rgb_ind_encoder_features_20261110/encoder_features.mp4) 和 [六时刻对照图](../../outputs/insert_plug/attention/act_insert_plug_v1_rgb_ind_encoder_features_20261110/encoder_features_overview.png)。来源是原 RGB ACT 最终030000、pos6、seed60、时序集成系数0.01、每次执行1步的19.0秒成功轨迹。工具复用该次成功执行保存的190帧真实观测和初始RNG，保持完整时序集成历史重算；全部执行动作和完整64步预测逐位一致，没有新增物理闭环试验。

```bash
python -m examples.insert_plug.policy.visualize_act_features \
  --source-run outputs/insert_plug/attention/act_insert_plug_v1_rgb_ind_ensemble_20261110 \
  --output outputs/insert_plug/attention/act_encoder_features_new \
  --device cuda --torch-threads 1
```

视频上排是双相机原图，中排是ResNet layer4特征幅值，下排是1×1投影后的视觉token幅值；后者尚未经过Transformer编码器混合。两者都来自真实前向的512×8×8特征，按空间位置计算通道RMS，不做空间softmax。ACT没有此前Diffusion的SpatialSoftmax关键点层，白框仅标最大RMS网格，不表示物体检测或学习到的关键点。幅值不等于动作注意力、相机重要性或因果贡献。

每层在全部190帧、两相机间共用一个固定色标，默认上限取99.5百分位；两层独立定标。显示可饱和，但 `features/` 中的原始特征与RMS不裁剪。`report.json` 记录方法、时序、尺度和源文件哈希，`feature_locations.csv` 记录峰值网格与幅值，`independent_validation.json` 独立重算380张投影和全部RMS并验证视频帧数。共享提取器 `src/mujoco_tools/act_features.py` 只读捕获真实backbone及投影输出；与既有注意力提取器合计26项相关测试通过。

### 遮挡相机，测量动作敏感性

注意力不直接等于输入对动作的影响。`camera_ablation` 在执行过程中保存的相同图像/状态上，分别遮挡 head（第三视角）和 wrist，比较模型预测；每组比较保持机器人状态、另一相机及历史相同，不会推进策略动作队列或改变原始执行记录。

```bash
conda activate mujoco
python -m examples.insert_plug.policy.camera_ablation \
  --source-run outputs/insert_plug/attention/act_insert_plug_v1_rgb_ind_ensemble_20261108 \
  --seed 20261108 --patch-repeats 3 --drop-probability 0.5
```

输入必须是每个控制步都保存当前观测的注意力运行（`n_action_steps=1`）。默认输出到新的 `outputs/insert_plug/camera_ablation/<model>_insert_plug_<version>_<rgb|rgbd>_[ind|ood]_<ensemble|stateless>_<时间>/`，可用 `--output` 指定；已有目录拒绝覆盖。共享的独立预测/遮挡工具在 `src/mujoco_tools/policy_ablation.py`，任务实验入口在 `policy/camera_ablation.py`。默认 CPU 两线程，不占用训练 GPU。

每个观测有 11 个推理分支：无遮挡基线、分别对两相机整幅平均色填充、整幅黑色填充，以及各相机三次随机半幅遮挡。随机半幅遮挡在 8×8 网格中抽取 32 格；head/wrist 使用相同的像素网格位置，固定种子可复现。平均色来自该源图像的各 RGB 通道均值，会保留全局颜色信息；黑色填充是另一种严重缺失视角测试。`--patch-fraction` 可改遮挡比例，实际面积按整格取整。

报告分别给出以下量，避免时序集成掩盖当前图像的影响：

- **当前预测 query 0**：标准化动作 RMS，以及七关节目标差的 RMS/最大值（度）和夹爪开度目标差（毫米）。完整 64 步预测块另行统计，不当成当前动作。
- **同历史、单次遮挡后的集成动作**：只替换最新预测，其余历史完全相同。影响等于当前预测差乘以最新预测的集成权重；当前参数在窗口填满后该权重约为 1.12%。
- **连续随机遮挡历史**：以默认 50% 概率在每个时刻遮挡整路相机，head/wrist 使用相同随机时刻，分别维护独立的集成历史。所有分支仍读取同一条记录的状态/图像，因此衡量的是固定参考轨迹上的累积动作变化，不是执行遮挡动作后的闭环轨迹。

三次空间遮挡先在同一帧取平均，再按整段及每 5 秒统计 mean/median/p95/max。默认把最大关节目标差超过 1° 或夹爪目标差超过 1 mm 计作“明显变化”；阈值可调整，仅是描述性工程尺度，不代表统计显著性或任务失败。连续帧相关，不将它们视为独立试验计算显著性。相机遮挡也可能产生训练分布外图像，应结合不同遮挡/填充方式比较，不能直接推广成全部位置或任务阶段的相机重要性。

`summary.json` 保存协议、逐组统计、时间段统计和基线重建检查；`data/predictions.npz` 保存所有分支的完整标准化/物理动作预测、随机遮挡网格和时刻；`data/metrics.npz` 保存逐帧指标；`previews/` 保存遮挡示例。源观测按路径及 SHA256 引用，不重复生成数据集。先用原观测预测并通过独立的 LeRobot 时序集成器重建原执行动作，超过浮点容差会报错，防止分析不匹配的模型或执行历史。

可在已有 matplotlib 的环境中生成独立 PNG/SVG 对比图；该绘图入口不加载 MuJoCo 或模型，本机可用系统 Python：

```bash
/usr/bin/python3 examples/insert_plug/policy/plot_camera_ablation.py \
  --report outputs/insert_plug/camera_ablation/<运行目录>/summary.json
```

已完成当前 ACT v1 / 030000、pos0、时序集成 0.01 的 20 秒参考轨迹实验：200 个时刻、2,200 组预测，随机整路遮挡在 200 步中实际触发 99 次。结果在 `outputs/insert_plug/camera_ablation/act_insert_plug_v1_rgb_ind_ensemble_20261108/`，见[对比图](../../outputs/insert_plug/camera_ablation/act_insert_plug_v1_rgb_ind_ensemble_20261108/summary.png)、[遮挡示例](../../outputs/insert_plug/camera_ablation/act_insert_plug_v1_rgb_ind_ensemble_20261108/previews/masks_0100.png)和[完整报告](../../outputs/insert_plug/camera_ablation/act_insert_plug_v1_rgb_ind_ensemble_20261108/summary.json)。

| 指标（在相同时刻与无遮挡预测比较，整段均值） | 遮第三视角 head | 遮腕部 wrist |
|---|---:|---:|
| 整幅平均色遮挡：当前 query 0 七关节目标差 RMS | 5.02° | 7.22° |
| 整幅平均色遮挡：当前 query 0 夹爪开度目标差 | 1.96 mm | 22.81 mm |
| 50% 概率随机整路遮挡：集成后七关节目标差 RMS | 2.23° | 3.50° |
| 50% 概率随机整路遮挡：集成后夹爪开度目标差 | 2.82 mm | 8.02 mm |

相机影响随时间、动作维度变化，不能简单指定一个固定“最重要视角”。整幅平均色遮挡时，前 0–5 秒当前关节目标差 RMS 为 head 3.44° / wrist 23.14°；5–20 秒则为 head 5.55° / wrist 1.91°，后段关节目标对第三视角更敏感。这个后段排序在黑色整幅遮挡和随机半幅遮挡下也一致；腕部遮挡对夹爪的整段影响更大。这些数值是固定参考状态上的目标变化，不是实际机械臂偏移，也不证明遮挡后的成功率；原 pos0 轨迹未完成插入。

已独立复算全部物理单位指标及两种集成历史，重现随机网格/遮挡时刻并核对源文件哈希。无遮挡基线与保存动作最大差约 2.4×10⁻⁷ rad、1.5×10⁻⁸ m，仅浮点误差；原数据与 checkpoint 配置未变，99 项相关测试通过。

### 历史部署验证

此前旧模型的 `030000` 在 pos0 的单条部署验证结果（不代表当前 50 步设置或新 v1 模型）：

| 执行长度 | 设备 | 结果 |
|---|---|---|
| 1 | CPU | 60 秒超时，停留在初始姿态附近，零违规接触 |
| 100（checkpoint 原值） | CPU | 9.707 秒出现插销与源孔壁违规接触，停止 |
| 10（历史设置） | CUDA | 接近并夹住零件，未继续拔出；60 秒超时，零违规接触 |

结果目录分别为 `outputs/insert_plug/rollouts/act_030000_pos0_n1_cpu/`、`act_030000_pos0_n100_cpu/`、`act_030000_pos0_n10_cuda/`。GPU 运行检查了完整 60,000 个物理步，双视角视频为 512×256、10 fps；60 秒仿真在未开启实时节流时约耗时 51 秒。最早 CPU 烟测的 XML 是根据报告保存的完整配置补建，并在报告中注明；后续运行在启动时直接保存实际 XML。三个设置均未完成任务，这是已测 pos0 的结果，不能推广成全部十个位置的成功率。

完整无窗口批量测试按模型、版本、模态、ind/ood 和测试时间保存。`outputs/test/act_insert_plug_20261107_2044/` 保留旧严格接触模式的 50 步批次，每个 pos 测试 10 次，共 100 次；更早的 10 步执行批次及其额外诊断已按用户要求删除。保留批次的历史结果为 **3 次完整成功、97 次接触阈值中止、0 次超时**，成功均来自 pos2（3/10）。原 `summary.json`、`episodes.csv` 和视频不改写；97 次中止只能说明触发了当时阈值，不能证明允许继续执行后仍会失败，也不能代表当前默认模式的成功率。

关闭接触中止后另测 pos3、seed 30 一次，完整执行 600 个动作、60 秒，无接触中止或运行异常，但超时未完成任务，见[单次新模式报告](../../outputs/test/act_insert_plug_20261107_2044_contact_response/pos3/summary.json)与[双视角视频](../../outputs/test/act_insert_plug_20261107_2044_contact_response/pos3/episode_000000/rollout.mp4)。视频 601 帧完整解码。这是单次检查，不是 100 条重新评估结果。

后续新模式批量测试使用新目录，例如下列命令；示例不表示已重新执行 100 次测试：

```bash
python -m examples.insert_plug.policy.evaluate --workers 5 --n-action-steps 50 \
  --output outputs/test/act_insert_plug_20261107_2044_contact_response_batch
python -m examples.insert_plug.policy.summarize \
  --root outputs/test/act_insert_plug_20261107_2044_contact_response_batch \
  --checkpoint outputs/train/act_insert_plug_20261107_2044/artifacts/checkpoints/030000 \
  --n-action-steps 50 --write
```

历史 pos0 逐步诊断发现，完整预测里超过 5 mm 的抬升在第 23–28 个动作才出现，前 10 步主要保持抓取，因此反复重新预测会推迟抬升。这是调整到 50 步的依据。当前每步仍为 100 ms；之后按用户要求关闭了默认附加接触中止，模型完成率需在新模式下另行评估。

## RGB-D 模型的 3 cm 平移响应实验

### 已验证成功原点的对照实验

根据用户对实验设计的修正，另完成了[成功原点对照报告](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_verified_grasp_20261110_162721/report.txt)：仅 **10 条**，两个模型共用此前各自 10/10 成功的 pos2，采用 seed20、每次执行32步后重规划。先执行两个原点，确认螺丝实际提起，再运行源底座与螺丝一起沿 ±X/±Y 平移30mm的八个条件。目标底座、机械臂初态、相机和物理参数保持一致；配对完整随机流。每条固定观察12.8秒。

原点要求左右夹指连续接触200ms、螺丝尖端相对TCP漂移≤0.2mm，并在尖端高于源底座口沿5mm后继续满足200ms稳定夹持。ACT / DP 在 **8.826 / 8.684秒**通过，见[原点验证图](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_verified_grasp_20261110_162721/plots/baseline_grasp_qualification.png)。八个平移条件均未在12.8秒内验证提起，这不代表无限延长后的最终任务结果。

在原点确认抓取所在控制周期末（6.4秒），沿物体平移方向的TCP差如下；理想响应为+30mm。负数是两个条件间的反向目标偏移，并非机械臂单条轨迹的运动方向。

| 平移 | ACT FK / 实际响应 mm | DP FK / 实际响应 mm | 训练源坐标包围盒 |
|---|---:|---:|---|
| +X | −5.28 / −5.37 | −5.94 / −5.99 | 外 |
| −X | 15.18 / 10.17 | 8.21 / 8.13 | 内 |
| +Y | 7.19 / 17.39 | 4.60 / 16.21 | 外 |
| −Y | 22.97 / 28.35 | −5.85 / −5.94 | 内 |

见[完整响应曲线](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_verified_grasp_20261110_162721/plots/translation_vs_time.png)和[抓取阶段比较](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_verified_grasp_20261110_162721/plots/grasp_translation_response.png)。ACT −Y 的实际同向响应虽接近30mm，但TCP距新源位置仍有7.45mm的XY误差，未能提起，说明还需看正交误差与实际抓取。坐标包围盒内不等于严格插值；单一位置和种子不能估计整体成功率，也不能唯一归因于内部空间表示。

新工具为 `policy.grasp_translation_probe`、`policy.grasp_translation_metrics`、`policy.plot_grasp_translation` 和 `policy.summarize_grasp_translation`。启动命令保存在 `outputs/_checks/rgbd_pos2_verified_grasp_3cm_20261110/run.sh`；重跑必须换新输出目录。独立审计通过全部10条轨迹、1290帧视频、128010个物理采样；FK重算误差0，chunk执行前缀完全一致，28个受保护文件未变，42项相关测试通过。

### 成功原点上的初始64步绝对坐标与第三视角对比

补充图保存在上述目录的 `full64_initial/`：[四方向绝对坐标](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_verified_grasp_20261110_162721/full64_initial/plots/absolute_shift_axis_64.png)、[完整XYZ](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_verified_grasp_20261110_162721/full64_initial/plots/absolute_tcp_xyz_64.png)、[配对位移响应](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_verified_grasp_20261110_162721/full64_initial/plots/translation_vs_horizon_64.png)。均值和最大/最小值按同一位置、同一query、同一坐标在种子间计算。ACT直接复用确定性预测，n=1、三者重合；DP对原点和四方向各做5次纯推理（种子20–24），阴影是实际采样极值，不是置信区间。未增加物理执行实验。DP保持n32设置，在切片前被动记录完整64步；seed20前32步和完整随机流均与原实验逐项一致。这里q32–63是初始预测的未执行部分，不能与上文重规划后的6.4秒结果混用。逐query的完整数值见 `pointwise_absolute_tcp_statistics.csv`。

[真实第三视角对比图](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_verified_grasp_20261110_162721/full64_initial/visuals/third_person_translation_comparison.png)依次展示原点和四方向的原始256×256输入、同一固定75×75区域放大、统一量程RGB差异图。原始PNG逐像素保真，没有重新渲染。螺丝头在+X、−X、+Y、−Y方向分别投影移动11.19、10.04、19.68、19.68像素。整幅图仅约1.03–1.59%的像素达到任一通道差≥5/255，主要因为背景不变；局部移动清楚可见。不能据整幅图相似直接归因于模型看不出位置变化，模型还有腕部RGB和米制深度输入。补充实验的独立审计复算了全部1920个FK目标、统计量、图像差异和投影，误差0且35个源文件未变。

### 早期初始 chunk 敏感性诊断

以下旧实验没有逐个验证成功原点，只能解释初始预测对位置变化的敏感性，不能代替上面的成功原点对照。

已完成的[实验报告](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_initial_open_loop_20261110_145234/report.md)包含 10 个布局、ACT 40 对与 DP 200 对比较、300 条完整开环执行，独立审计通过。螺丝与源底座移动 30 mm 后，首次闭爪命令处的平均同向 FK 响应为 ACT **9.69 mm**、DP **4.95 mm**；仅统计源位置逐轴训练范围内的条件，分别为 **16.64 mm**、**8.63 mm**。完整 64 步终点的 FK 响应为 **9.72 / 5.00 mm**，实际末端响应为 **12.38 / 7.81 mm**。这些是两个条件间的位置差，不是单条轨迹总路程或抓取成功率。见[响应曲线](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_initial_open_loop_20261110_145234/plots/translation_vs_horizon.png)和[预测与实际执行对比](../../outputs/insert_plug/translation_probe/act_diffusion_insert_plug_v1_rgbd_initial_open_loop_20261110_145234/plots/predicted_vs_executed.png)。

`policy.translation_probe` 将**螺丝与源底座一起**沿世界坐标 ±X、±Y 平移 3 cm，固定机械臂初始关节、夹爪、目标底座、相机、外观和物理配置，重新渲染双路 RGB 与米制深度。采用已完成训练的 ACT / Diffusion RGB-D 最终 `030000` 模型，保留其预处理、反归一化和完整 64 步预测。ACT 只需一组确定性预测；Diffusion 默认使用五组配对种子，每个条件重置全部 CPU/CUDA RNG，覆盖 DDPM 每一步的随机噪声。

动作是绝对关节目标，不能累加关节角。工具对每条命令单独做 FK，比较平移前后的末端目标差，记录沿物体平移方向的有符号位移、增益、正交偏移和三维残差。逐步笛卡尔位移累加等于末端终点减初点。固定 query31、query63、基准首次闭爪 query 和两侧各自首次闭爪 query 分开报告；连续两条夹爪命令 ≤26 mm 仅是意图闭爪的代理，并不证明抓取成功。

默认还以原有 10 Hz 指令、1 kHz 物理执行全部 64 步，单独保存实际 TCP 轨迹，不修改、裁剪或纠正模型动作。这是初始 chunk 的 6.4 秒开环诊断；采用 32 步重规划的部署结果可能不同。未完整执行的回放显式记录错误，不计算完整回放指标。训练坐标范围内外的平移分别统计，坐标范围内也不等于训练布局凸包内。

在工作区根目录运行，输出必须是新目录：

```bash
conda activate mujoco
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m examples.insert_plug.policy.translation_probe \
  --act-checkpoint outputs/train/act_insert_plug_v1_rgbd_20261110_005600/artifacts/checkpoints/030000/pretrained_model \
  --diffusion-checkpoint outputs/train/diffusion_insert_plug_v1_rgbd_20261110_005600/artifacts/checkpoints/030000/pretrained_model \
  --device cuda --backend egl --seeds 0 1 2 3 4 \
  --output outputs/insert_plug/translation_probe/new_run
# 绘图可用现成的系统 Python，推理与物理仍使用共享 mujoco 环境。
/usr/bin/python3 -m examples.insert_plug.policy.plot_translation_probe \
  --output outputs/insert_plug/translation_probe/new_run
python -m examples.insert_plug.policy.summarize_translation_probe \
  --output outputs/insert_plug/translation_probe/new_run
```

CPU 环境可改为 `--device cpu --backend osmesa`。原始 RGB-D、场景 XML、预测动作、配对 RNG、FK 与实际轨迹、汇总及 PNG/PDF 图保存在输出目录；checkpoint 和数据集不改写。位移响应不足能说明当前策略没有充分跟随源物体位置，但不能单凭该实验将原因唯一归结为内部空间表示，也不能证明点云或位姿输入必然改善成功率。

## STL 与碰撞几何

```bash
python -m mujoco_tools.assets.inspect_stl /path/to/part.stl --units mm
python -m examples.insert_plug.environment.prepare_objects
```

两个原始 STL 均保留不变。螺栓的杆与六角头分开构建凸碰撞体；放置桶按角度和高度分解为多个凸碰撞体，保留孔腔、内底与桶口。完整 STL 用于外观，避免把带孔桶直接当成一个会封住孔腔的凸包。派生资产目录记录生成方法与几何近似；修改 STL 后需要重新生成碰撞资产并核对配置尺寸。
