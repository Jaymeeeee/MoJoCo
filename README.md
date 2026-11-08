# MuJoCo 仿真工作区

仿真 Conda 环境：`mujoco`；数据转换独立使用 `lerobot`；ACT 策略部署使用 `mujoco_policy`。通用资源放在顶层，每个场景的实现放在 `examples/`。

```text
mujoco/
├── assets/                  # 共享机器人、网格与材质
├── config/                  # 场景参数和清理策略
│   ├── insert_plug.yaml
│   ├── insert_plug_positions.yaml # 10 组已检查的 2 cm 网格位置
│   └── cleanup.yaml
├── examples/
│   └── insert_plug/        # scene.xml、场景构建、控制器和采集逻辑
├── src/mujoco_tools/        # 通用路径、环境诊断、产物清理
├── scripts/                 # 资产下载、STL 检查、清理入口
├── tests/                   # 自动化测试
├── outputs/
│   ├── datasets/insert_plug/ # 每批一个目录，每条独立子目录内保存四个文件
│   ├── insert_plug/       # 按需生成的图片、视频和计划导出
│   └── _checks/            # 临时测试产物，自动清理
├── environment.yml
└── pyproject.toml
```

## 运行场景

```bash
conda activate mujoco
cd ~/Projects/mujoco
python -m examples.insert_plug view
```

`view` 只预览静态场景，可以旋转、缩放视角；不会运行专家、推进物理或采集数据。关闭窗口后临时场景文件自动删除。`--camera third_person` 或 `--camera wrist` 可切换初始相机视角。

当前场景使用 FR3、Robotiq 2F-85 和用户提供的螺栓/放置桶 STL。保存当前初始场景图片可运行 `python -m examples.insert_plug snapshot --backend egl`，图片位于 `outputs/insert_plug/scene_preview/`。

当前配置生成的完整场景保存在 `examples/insert_plug/scene.xml`。修改配置后运行 `python -m examples.insert_plug build` 重新生成。采集目录中的 `episode_<seed>.xml` 则是该条轨迹实际使用的场景快照，可能包含随机化后的孔位，需随 HDF5 一起保留。

第三视角对外字段为 `observation.images.head`，腕部为 `observation.images.wrist`。当前默认同时采集 `observation.depth.head`、`observation.depth.wrist`，保存为与 RGB 对齐的 float32 米制深度；开关在 `config/insert_plug.yaml` 的 `cameras.depth.enabled`。`snapshot` 还会生成原始深度 NPY、米制深度预览和相机标定 JSON。`view --camera head` 可查看固定第三视角。旧记录仍可读取和回放；深度输入 ACT 的训练适配需另行接入。

先导出完整名义计划而不执行动作、推进物理或采集图像：

```bash
python -m examples.insert_plug plan --output outputs/insert_plug/plans
```

它保存 `nominal_trajectory.json` 与 `nominal_trajectory.npz`。两个放置桶固定在世界坐标系中，没有关节，不会被接触推走；平面位置只在 `config/insert_plug.yaml` 的 `board.source_xy` 和 `board.target_xy` 中配置。规划采用任务约束下的平滑笛卡尔轨迹与 IK：默认提起使用余弦速度加减速与匀速段，抓取点、放置点上方采用具有连续向下速度的五次 Hermite 连接，并检查可达性及运动界限。默认不做附加碰撞阈值预检；MuJoCo 仍通过原碰撞体计算实际接触、摩擦和阻挡。

执行、采集并生成双相机视频：

```bash
python -m examples.insert_plug demo --backend egl --width 512 --output outputs/datasets/insert_plug/under20_demo --video
```

`demo` 和 `collect` 自动先求解完整名义计划，再执行动作。默认 `continuous_hover=true`：通过抓取点上方时以约 20 mm/s 继续下降，通过目标上方及预插入位置时以约 8 mm/s 连续下移，不在这些路点停稳或重新从零速起步。目标上方仍会测量实际夹持变换，但后续命令衔接上一条目标及其速度，避免退回滞后的实际位置。插入前即时检查孔轴匹配；默认连续模式不执行停住后再纠偏的循环，条件不满足会报告失败。

搬运 TCP 高度从桌面上方 16 cm 降为 10 cm；抓取前接近高度为抓取点上方 4 cm，释放后撤离 5 cm。提起使用 `cosine_velocity`，时长 4 秒，两端加减速各占 30%，适配缩短后的约 56 mm 行程。下降抓取、搬运、对准、浅插的配置时长分别为 1.0、1.5、1.2、2.5 秒，规划器可为满足运动界限延长时长。空载接近允许 0.24 m/s，持物段笛卡尔速度上限为 0.18 m/s；关节速度、加速度上限为 1.4 rad/s、4 rad/s²。浅插深度仍为 12 mm，松爪时长仍为 1 秒。控制和记录均为 10 Hz，每个动作保持 100 ms。

实际抓取与浅插终点仍需精确到位，双指接触稳定至少 200 ms、漂移合格后再提起；最终落座仍检查至少 250 ms 的稳定性。默认 `task.planning.strict_collision_checks=false`，不创建 `MotionGuard`，普通抓取、孔壁和孔口接触不会触发附加阈值中止。执行结果与检查信息存入 HDF5 元数据 `execution_result`；需要导出名义计划时单独运行 `plan`。

上述命令的数据集保存在 `outputs/datasets/insert_plug/under20_demo/`，请求的双相机视频另存到 `outputs/insert_plug/previews/under20_demo/preview.mp4`。两侧标注 Head (third-person) 和 Wrist，独立页眉显示仿真时间与当前阶段，按记录帧率播放并保留完整原始画面。已有 episode 和视频不会覆盖，重复执行请改用新输出目录。

当前默认按“完整任务少于 20 秒”调整，在原十个固定布局各验证一次，**10/10 完成**，共 1,899 个控制步，时长 **18.6–19.7 秒、平均 18.99 秒**。十个布局均无额外对准纠偏，抓取点上方、目标上方及预插入边界没有重复保持指令。持物 TCP 峰值约 10.4 cm，搬运时插销尖端与孔口间距约 2.6 cm 以上。见[当前验证报告](outputs/_checks/insert_plug_under20/summary.json)和[双视角视频](outputs/insert_plug/previews/under20.mp4)。这是已测十个布局的结果，改变布局后仍需验证。

场景 YAML 的默认固定位置另已完成一条 **18.9 秒、189 帧、10 Hz** 的双路 RGB-D 实录，第三视角 MP4 审计通过。按保存动作独立回放后任务成功，关节与插销状态相对原记录偏差均为零；双视角视频完整解码 189 帧，82 项回归测试通过。一次性原始试录文件清理，保留[录制审计](outputs/_checks/insert_plug_under20/recording_validation.json)、[动作回放报告](outputs/_checks/insert_plug_under20/recorded_action_replay.json)和预览视频。

上一版连续路点专家平均 24.42 秒，其[报告](outputs/_checks/insert_plug_continuous_hover/summary.json)与[视频](outputs/insert_plug/previews/continuous_hover.mp4)保留为历史对照。更早的平滑专家平均 30.3 秒，保留训练批次平均 38.41 秒。已有 100 条训练数据、LeRobot 数据、模型和策略测试结果均未改写；当前更新没有重新采集 100 条或重训 ACT。抓稳、对准、插入和最终稳定要求保留，未启用接触诊断时不声明“零违规接触”。

需要实时窗口时运行：

```bash
python -m examples.insert_plug demo --backend glfw --viewer --realtime
```

需要图形桌面才能弹出窗口。无窗口采集使用 EGL：

```bash
python -m examples.insert_plug collect --backend egl --episodes 10 --randomize-mm 0 \
  --output outputs/datasets/insert_plug/under20_fixed_v1
```

`demo` 和 `collect` 默认都保存到 `outputs/datasets/insert_plug/run_<时间>/`。每条轨迹有独立的 `episode_<seed>/` 子目录，内部保留同名 `.h5`、`.yaml`、`.xml`、`.mp4` 四个文件，例如 `run_<时间>/episode_000000/episode_000000.h5`。每条 MP4 默认生成，内容是原分辨率、不叠字的第三视角 `head` RGB，默认 10 fps；HDF5 仍包含两路 RGB 和深度。`--video` 额外请求的双视角视频另存到 `outputs/insert_plug/previews/<运行目录名>/preview.mp4`。`--no-images` 物理检查放到 `outputs/_checks/insert_plug/`，不生成 HDF5 或视频。显式 `--output` 相对于运行命令的当前目录。

历史固定位置测试批保留在 `outputs/datasets/insert_plug/test/`，10 条均已成功完成并通过校验，每条 384 帧、15 Hz。这些历史记录不会随当前 10 Hz 配置更新，重复采集应使用新的批次目录或不同 seed。HDF5 中的 `model_path` 与 `execution_result.model` 使用同目录 XML 的相对文件名，便于一起移动整条记录。

记录频率由 `config/insert_plug.yaml` 的 `recording.sample_hz` 或 `--sample-hz` 设置，默认 10 Hz；专家控制默认也为 10 Hz，物理仿真为 1 kHz。每条 `action` 在采样时下发，并在接下来的 100 个 1 ms 物理步中保持恒定，直到下一条观测。专家先将目标量化为 float32，再将这些精确数值交给仿真并保存到 HDF5；导出 LeRobot 的 float32 动作不会再次改变控制数值。时间戳来自真实物理时刻，默认为 `0.0、0.1、0.2…` 秒。若显式设置不同的采样和控制频率，应按记录元数据解释动作保持时间。`collect` 默认不随机化底座位置，需要随机布局时显式使用 `--randomize-mm`。

根目录 `data/peg_transfer/` 的旧记录已清理，旧 `outputs/peg_transfer/` 历史记录仍保留；旧资产目录名通过符号链接兼容历史场景。新记录使用 `insert-plug-v3`，仍支持读取与回放 `peg-transfer-v1/v2`。

已保存的记录和视频不会随模型修改而更新；完整规划范围、执行条件和数据格式见[insert_plug 场景](examples/insert_plug/README.md)。控制台命令 `insert-plug` 也可作为入口。

## 2 cm 网格位置批量采集

`config/insert_plug_positions.yaml` 保存 10 组固定位置。两个底座相对于基础配置的位置按 2 cm 整数格移动，当前范围为每轴 ±4 cm；每组的两个底座位置均与其他组不同。布局已检查实际 STL 视觉网格是否完整落在第三视角视锥内，并通过完整任务的物理预执行。

保留的历史训练批次使用先前专家，在相同 10 个固定布局各采集 10 条，曾替换旧的 50 Hz 控制／10 Hz 记录批次。其控制和记录均为 **10 Hz**、物理为 **1 kHz**，包含双路 RGB、float32 米制深度及第三视角 MP4；**100/100 条成功，共 38,410 帧、400 个文件，约 13.64 GB**。采集当时的独立审计和 100 次保存动作物理重放全部通过、零违规接触，关节和插销状态最大偏差均为零；初始和最终视觉网格的画面边缘最小余量为 51.694 px。这些是先前数据的结果，不是当前平滑专家新采的 100 条。

`config/insert_plug_positions.yaml` 已按当前专家配置重新验证，底座坐标保持不变。后续新采应使用独立目录，例如：

```bash
python scripts/collect_insert_plug_positions.py --workers 3 \
  --output outputs/datasets/insert_plug/under20_v1

# 只采集或继续检查一个位置
python scripts/collect_insert_plug_positions.py --position pos0 --workers 1 \
  --output outputs/datasets/insert_plug/under20_v1

# 采集完成后审计全部 100 条记录
python scripts/audit_insert_plug_batch.py --positions-config config/insert_plug_positions.yaml \
  --root outputs/datasets/insert_plug/under20_v1
```

历史数据仍位于 `outputs/datasets/insert_plug/pos0/` 至 `pos9/`；上述新命令将写入 `outputs/datasets/insert_plug/under20_v1/pos0/` 至 `pos9/`。全局编号从 0 到 99，例如 `pos0/episode_000000/`、`pos1/episode_000010/`、`pos9/episode_000099/`，每条仍保留同名 HDF5、YAML、XML、MP4 四个文件。原 `test/` 中的 15 Hz 固定位置记录保留。

批量脚本会校验并跳过已完成且配置相同的记录；不完整或配置不同的记录会保留并报错。基础配置的 SHA256 与位置清单不一致时会拒绝启动，需要重新生成位置清单并选择新的输出目录。新清单可用 `python -m examples.insert_plug.layouts --output config/insert_plug_positions_new.yaml` 生成，已有清单不会被覆盖。详细参数和目录示例见[场景说明](examples/insert_plug/README.md#网格位置批量采集)。

## LeRobot 数据集

先前专家采集的历史批次已合并到 **`outputs/datasets/insert_plug/lerobot/`**，包含 100 条、38,410 帧、10 Hz，约 11.65 GB，保持原样。它此前替换的是 50 Hz 控制的旧批次，不会因这次专家优化再次删除或更新。转换使用官方 LeRobot 0.7.0 的 `LeRobotDataset` API 写入 v3 格式，仅在本地保存。独立 `lerobot` 环境使用 Python 3.12 和 CPU PyTorch，不影响 `mujoco` 仿真环境。历史批次的全部动作、状态、时间戳、阶段及深度像素已逐帧校验，动作 float32 转换无损，最终目录读取验证通过。

```bash
conda activate lerobot
# 先完成 under20_v1 的十布局采集；目标转换目录须尚不存在
python scripts/export_insert_plug_lerobot.py \
  --source-root outputs/datasets/insert_plug/under20_v1 \
  --output outputs/datasets/insert_plug/under20_v1/lerobot
```

`observation.images.head/wrist` 保存为两路 RGB H.264 视频；`observation.depth.head/wrist` 保存为 Parquet 内嵌的 float32 TIFF，单位米，转换时与原 HDF5 逐像素校验。`observation.state` 和 `action` 各 8 维（7 个关节角 rad ＋夹爪开口 m）。`meta/export_manifest.json` 保存原 HDF5 路径、位置编号、帧范围、相机标定与动作语义；验证结果见 `meta/export_validation.json`。

ACT 输入配置片段位于 `config/act_insert_plug_features.json`，显式选择两路 RGB 和机器人状态。标准 ACT 不会自动利用这些米制深度；使用深度需要另加编码融合。当前数据的策略动作和专家目标均为 10 Hz；部署时每个动作执行一次 `Simulation.step(action)`，在 1 kHz 物理循环中保持 100 ms。读取示例、环境重建和输入配置见[场景 LeRobot 说明](examples/insert_plug/README.md#lerobot-转换与-act-输入)。

## 训练后部署 ACT

通用策略适配器在 `src/mujoco_tools/lerobot_policy.py`，当前支持 LeRobot ACT；场景执行入口在 `examples/insert_plug/rollout.py`。默认 checkpoint 和执行参数在 `config/insert_plug_policy.yaml`。模型配置、权重、输入归一化和动作反归一化均从完整的 `pretrained_model/` 加载，支持传入其上一级训练步目录。不会改写模型或训练数据。

```bash
conda activate mujoco_policy
cd ~/Projects/mujoco

# 桌面终端：模型闭环控制，显示实时窗口并保存双视角视频
python -m examples.insert_plug rollout --viewer --realtime

# 无窗口：评估训练布局 pos0，默认保存视频
python -m examples.insert_plug rollout --backend egl --position 0

# 十个布局分别评估一次
python -m examples.insert_plug rollout --backend egl --position all

# 替换为其他 ACT checkpoint，每次预测后连续执行 50 个动作
python -m examples.insert_plug rollout --backend egl \
  --checkpoint /path/to/checkpoints/030000 --n-action-steps 50
```

策略动作固定为 10 Hz：每个 8 维绝对目标（7 个关节角 rad、完整夹爪开度 m）保持 100 ms，物理子步为 1 ms。本任务默认 `n_action_steps=50`，依次执行 50 个动作、5 秒后重新预测；当前 checkpoint 的 `chunk_size=100` 保持不变。可用 `--n-action-steps 1` 每步重新预测，100 步则对应 10 秒。`--realtime` 只调整墙钟播放速度，不改变控制和物理时间；计算较慢时仿真会慢于实时。

默认使用位置清单中的 pos0；`--position config` 使用场景 YAML 的底座位置。`--episodes` 是每个选中位置的重复次数，默认 1；`--duration` 是每条最大仿真秒数，默认 60。`--device cpu/cuda/auto` 控制推理设备，`--no-video` 禁用视频。部署只输入两路 RGB 和状态，不把深度、目标位置或专家阶段传给该 ACT。

结果保存在 `outputs/insert_plug/rollouts/run_<时间>/`，包含执行报告、场景快照和双视角视频；显式 `--output` 必须指向尚不存在的目录。默认不扫描附加接触阈值、不因普通接触停止；成功按真实插入、松爪、撤离及连续至少 0.25 秒稳定判断。非法动作、非有限值和数值异常仍会停止并保存原因。`--contact-diagnostics` 可显式记录旧接触诊断但不否决任务成功；`--stop-on-contact` 才启用旧严格接触中止模式，旧 `--continue-on-contact` 保留兼容。诊断关闭时 `contact_clean=null`、接触检查次数为 0，表示未检测，不代表没有接触。全部任务成功返回 0，否则返回 2；策略成功率需要独立评估。

更多接口、环境重建及结果字段见[场景部署说明](examples/insert_plug/README.md#act-策略部署)。

此前 `030000` checkpoint 的 pos0 单条验证中，1 步执行长度在初始位置附近超时；100 步在 9.707 秒因插销碰到源孔壁而停止；10 步在 CUDA 上能接近并夹住零件，但未继续拔出，60 秒超时且无违规接触。这些历史单条结果不代表当前 50 步设置的表现。独立的 10 步单条[双视角视频](outputs/insert_plug/rollouts/act_030000_pos0_n10_cuda/episode_000000/rollout.mp4)与[报告](outputs/insert_plug/rollouts/act_030000_pos0_n10_cuda/episode_000000/result.json)仍保留。

## 模型批量测试

训练模型的完整无窗口测试使用 `outputs/test/`，与 `outputs/train/` 对应，按训练运行名称和位置分组：

```bash
conda activate mujoco_policy
python scripts/test_insert_plug_policy.py --workers 5 --n-action-steps 50 \
  --output outputs/test/act_insert_plug_20261107_2044_contact_response_batch
```

该入口使用 `config/insert_plug_policy.yaml` 的模型和动作执行长度，在 pos0–pos9 各实际运行 10 次；每次重新初始化场景、清空策略队列，按位置分配不同随机种子但不随机改变布局。每个 `posN/episode_000000` 至 `episode_000009` 保留 `result.json`、`scene.xml`、双视角 `rollout.mp4`。`eval_config.json` 保存本批参数、冻结的配置和模型文件校验值，`launch_summary.json` 保存进程完成情况。已有输出目录拒绝覆盖。固定初始条件的重复结果用于检查重复性，不能等同于 100 种随机场景的泛化测试。

完成后独立核验全部报告和视频，生成根目录的 `summary.json`、`episodes.csv`：

```bash
python scripts/summarize_insert_plug_policy_test.py \
  --root outputs/test/act_insert_plug_20261107_2044_contact_response_batch \
  --checkpoint outputs/train/act_insert_plug_20261107_2044/artifacts/checkpoints/030000 \
  --n-action-steps 50 \
  --write
```

启动器完成全部物理测试与模型全部成功是两个不同指标；模型超时、运行异常或显式严格模式下的接触停止也会保留结果。默认汇总统计物理成功与停止原因；接触诊断关闭时接触相关字段为未检测，不把它当作失败或“零碰撞”。上述新目录示例不会覆盖旧批次，也不表示已执行新一轮 100 次测试。

旧的 10 步执行批次（0 次完整成功、72 次接触停止、28 次超时）及其诊断文件已按用户要求删除。`outputs/test/act_insert_plug_20261107_2044/` 保留随后运行的 **50 步、旧严格接触模式批次**：pos0–pos9 各 10 次、10 Hz、每条最多 60 秒，全部 100 次已结束。历史结果为 **3 次完整成功、97 次接触阈值中止、0 次超时**；成功均来自 pos2（3/10）。这些中止只表明触发旧阈值，不能证明允许继续执行后仍会失败，也不能用 3% 代表当前默认接触模式的成功率。原 `summary.json` 和 `episodes.csv` 不作改写。[历史成功示例视频](outputs/test/act_insert_plug_20261107_2044/pos2/episode_000001/rollout.mp4)仍可查看。

关闭接触中止后，另对 pos3、seed 30 做了单次检查：完整执行 600 个动作、60 秒，没有接触中止或运行异常，但超时未完成任务。[单次报告](outputs/test/act_insert_plug_20261107_2044_contact_response/pos3/summary.json)与[双视角视频](outputs/test/act_insert_plug_20261107_2044_contact_response/pos3/episode_000000/rollout.mp4)独立保存，视频 601 帧完整解码；这不代表十布局重新评估后的成功率。

历史诊断显示，pos0 抓取后的完整预测包含约 110 mm 的未来抬升，首次超过 5 mm 的抬升位于第 23–28 个动作；只执行前 10 步就重新预测会反复推迟后段抬升。当时实际下发动作与被选出的预测逐值一致，未见力矩饱和。这是调整为 50 步的依据，但较长执行段仍须重新验证接触和任务成功率。

## 环境与工具

```bash
# 仅在重建环境时执行
conda env create -f environment.yml
conda activate mujoco
python -m pip install -r requirements-lock.txt
python scripts/fetch_robot.py
python scripts/fetch_gripper.py

# 日常检查
python -m mujoco_tools.doctor --backend egl
python scripts/inspect_stl.py /path/to/part.stl --units mm
python -B -m pytest -q
```

环境激活时临时清空 `PYTHONPATH`，避免系统 ROS 2 的 Python 包混入。此项目按 editable 工作区安装，资产和配置直接使用顶层目录；后续场景放在 `examples/<场景名>/`。

## GitHub 与其他电脑迁移

主仓库保存代码、配置、脚本、环境定义、依赖锁文件及完整的 `assets/`，包括用户提供的两个 STL。`outputs/` 已被 Git 忽略，数据集、checkpoint 和测试视频单独传输，保持原有目录结构。上传或克隆主仓库不会复制这些输出，也不会复制已安装的 Conda 环境。

`third_party/lerobot` 使用 Git 子模块，来源为 `https://github.com/Jaymeeeee/lerobot.git`。首次迁移固定在 `d490e472a5d1a207cfe0255caaaf74b6d66e4656`；实际使用版本以主仓库提交保存的子模块指针为准。保留第三方资产及代码原有的许可证文件。

另一台电脑安装 Git 和 Conda 后执行：

```bash
git clone --recurse-submodules https://github.com/Jaymeeeee/MoJoCo.git
cd MoJoCo

# 如克隆时没有带 --recurse-submodules，补运行：
git submodule update --init --recursive

# 仿真与专家采集
conda env create -f environment.yml
conda activate mujoco
python -m pip install -r requirements-lock.txt
python -m examples.insert_plug build
python -m examples.insert_plug view
```

当前生成的 `examples/insert_plug/scene.xml` 包含生成机器上的绝对资产路径，迁移后须运行上述 `build` 重新生成。`view` 需要图形桌面；无窗口环境可先用 `python -m mujoco_tools.doctor --backend egl` 检查渲染环境。

需要转换数据时，另建 `lerobot` 环境：

```bash
conda env create -f environment-lerobot.yml
conda activate lerobot
python -m pip install -r requirements-lerobot-lock.txt
```

需要部署 ACT 时，在项目根目录运行：

```bash
bash scripts/setup_policy_env.sh
conda activate mujoco_policy
```

部署锁文件针对 Linux x86_64、Python 3.12 和 CUDA 12.6；不同操作系统、CPU 架构或显卡需要另行适配，显卡驱动不随 Git 或 Conda 环境迁移。仅推理需要 checkpoint 中完整的 `pretrained_model/`，包括模型权重、配置和归一化统计；继续训练还需保留 `training_state/`。按 `config/insert_plug_policy.yaml` 指定的相对目录恢复 checkpoint，或运行时传入 `--checkpoint`。数据集、历史测试结果按需恢复到 `outputs/` 中原来的位置，避免覆盖已有批次。

日常同步代码和配置：

```bash
git add .
git diff --cached --stat
git commit -m "Describe the change"
git push
```

`git add .` 会遵守忽略规则；不要使用 `git add -f outputs`。如果修改 LeRobot 子模块，须先在子模块仓库提交并推送，再在主仓库提交新的子模块指针。其他电脑更新时执行 `git pull --recurse-submodules`，再运行 `git submodule update --init --recursive`。

## 测试产物清理

临时结果统一进入 `outputs/_checks/`，不要在顶层写迁移报告、诊断截图或测试 JSON。一次性验证优先使用系统临时目录，工作结束后删除。

已配置用户级 `mujoco-test-cleanup.timer`，每天检查并删除 **超过 7 天**的临时测试产物；离线错过的检查会在用户服务恢复后补执行。`outputs/datasets/`、历史 `data/`、`assets/` 和其他正常输出不在清理范围内。

```bash
python scripts/cleanup_outputs.py          # 预览
python scripts/cleanup_outputs.py --apply  # 按保留规则清理
systemctl --user status mujoco-test-cleanup.timer
```

保留期限见 `config/cleanup.yaml`。清理器不跟随符号链接，范围固定为 `outputs/_checks`。
