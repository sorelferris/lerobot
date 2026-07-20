# FUXI 自研子系统说明

本目录基于 LeRobot 官方仓库演进，二次开发集中在**异步推理框架、Value/ACP 训练流水线、SO101 遥操作、数据集工具链**四条主线。下文只列**自研模块**（已排除 `lerobot-train` / `lerobot-record` / `lerobot-calibrate` / `lerobot-edit-dataset` 等官方命令的薄壳脚本）。

> 阅读顺序建议：先看「〇、重点入口」理解 `train_pi05.sh` 与 `serve_policy.sh` 的定位，再分线读「一、异步推理」→「二、Value + ACP」→「三、数据集工具」。

---

## 〇、重点入口（必读）

下面两个脚本虽本身是薄壳，但**承载了项目最常用的两条工作流**：pi05 策略训练、Policy Server 启动。所有下游模块（ACP 训练、在线推理、Value 打分）都依赖它们的前置产物。二次开发者修改或排错时优先看这一节。

### 0.1 `scripts/train_pi05.sh` — pi05 策略训练入口

**角色**：包装官方 `src/lerobot/scripts/lerobot_train.py`，把 pi05 base 模型的微调参数固定成项目内的标准配方。执行产物 `sorel/pi05-<dataset>` 是 ACP 训练（`train_acp.sh`）和推理（`serve_policy.sh pi05`）的输入。

**关键项目化配置**（区别于裸跑 `lerobot-train` 的地方）：

| 项 | 值 | 用意 |
| --- | --- | --- |
| 预训练底座 | `lerobot/pi05_base` | 固定升级路径，避免下游实验被底座版本漂移影响 |
| `NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1` | `1` | RTX 4000 系列 GPU 规避 P2P/IB NCCL 报错 |
| `policy.gradient_checkpointing=true` | true | 显存换时间，适配单卡训练 |
| `policy.compile_model=false` | false | 关闭 torch.compile，与 pi05 VLM 路径不兼容 |
| `policy.dtype=bfloat16` | bfloat16 | 数值稳定性 + 显存占用平衡 |
| `policy.freeze_vision_encoder=false` / `train_expert_only=false` | false | VLM 全量参与训练（ACP 训练依赖 value-conditioned action expert） |
| `rename_map` | `front/wrist → camera1/camera2` | 与项目内数据集命名约定对齐，配套 `infer.sh` 使用 |
| `batch_size=32` / `steps=50_000` | 固定 | 项目经验值，二次开发时按需改 |
| `wandb.enable=true` | true | 默认开启 wandb 日志 |

**用法**：

```bash
./scripts/train_pi05.sh <dataset>          # 必传 1 个参数：数据集名（实际 repo 为 sorel/<dataset>）
```

> 衍生：`finetune_pi05.sh` 是它的「精简版」，差异在于 `HF_ENDPOINT="https://hf-mirror.com"`（国内下载底座）与固定 `output_dir`，用法同 `train_pi05.sh`。

---

### 0.2 `scripts/serve_policy.sh` — Policy Server 一键启动

**角色**：包装自研的 `scripts/tools/policy_server.py`（ZMQ 异步推理服务端），是项目内唯一推荐的「加载 checkpoint → 起 server」入口。下游 `infer.sh`（在线推理）和 `replay.sh`（回放评估）都依赖它提供的端口。

**关键项目化配置**（区别于裸跑 `policy_server.py` 的地方）：

| 项 | 值 | 用意 |
| --- | --- | --- |
| `--host=0.0.0.0` | 全部网卡监听 | 支持容器外/同网段多机访问 |
| `--policy_device=cuda` | GPU | 服务端推理设备；与 `client_device` 解耦 |
| `--actions_per_chunk=50` | 50 | 与 `train_acp.py` 训练时的 chunk 长度保持一致，避免 client 端 chunk 截断 |
| `--rename_map` | 见脚本内示例 | 兜底映射摄像头 key；当前默认空 `{}` 表示无重命名 |
| **端口自动分配** | `port = 9000 + CUDA_VISIBLE_DEVICES`（无 `CUDA_VISIBLE_DEVICES` 时回退 `9000`） | 多卡部署时端口自动错开，配合 `policy_dashboard.py` 实现零配置批量管理 |

**用法**：

```bash
./scripts/serve_policy.sh <policy_type> <pretrained_name_or_path> [port]
# 例：./scripts/serve_policy.sh pi05 outputs/train_acp/record_0429/checkpoints/last
# 例：省略 port → 自动按 CUDA_VISIBLE_DEVICES 派生
```

**上下游关系**：

```
train_pi05.sh  ──产物 sorel/pi05-<ds>──►  serve_policy.sh pi05 …  ──►  infer.sh / replay.sh
                                            ▲
                                  port 由 CUDA_VISIBLE_DEVICES 派生
                                            │
                                  policy_dashboard.py 可视化批量管理
```

> 修改提示：① 改 `actions_per_chunk` 时**必须同步** `train_acp.py` 的 chunk 设置；② 端口默认值由 `CUDA_VISIBLE_DEVICES` 控制，多 GPU 部署前先确认环境变量；③ `--rename_map` 默认空 dict 表示无重命名，按需打开注释启用 `right → right_camera` 等映射。

---

## 一、异步推理子系统（Policy Server / Client）

ZMQ + 多 GPU 端口分配，支持在线、离线、Replay、Replay-with-Value 四种调用形态，并附带 TUI 仪表盘。

| 路径 | 角色 | 关键能力 |
| --- | --- | --- |
| `scripts/tools/policy_server.py` | `PolicyServer` | 加载策略、检查点校验、warmup、动作块预测、observation/action 协议；支持 `actions_per_chunk` 和 `rename_map`。入口 `serve_policy()`。 |
| `scripts/tools/policy_server_v1.py` | 服务端 V1（旧实现） | 保留作为历史参照，新部署用上面那份。 |
| `scripts/tools/policy_client.py` | `PolicyClient` + `MockRobot` | 异步动作请求、chunk 合并（`weighted_average` / `latest_only`）、防止并发 observation 的 in-flight 锁。 |
| `scripts/tools/policy_dashboard.py` | `PolicyDashboardApp` (Textual) | 自动扫描 `outputs/` 下的 checkpoint、按 `CUDA_VISIBLE_DEVICES` 自动分配端口、批量启停 server、实时日志面板。 |
| `scripts/tools/infer.py` | 在线推理主循环 | 摄像头 → 观测 → `PolicyClient` → 写 `LeRobotDataset` → Rerun 可视化。 |
| `scripts/tools/infer_offline.py` | 离线推理 | 回放 dataset episode，用 `MockRobot` 拉取策略动作，与遥操作动作对比，支持远程 Rerun。 |
| `scripts/tools/infer_online.py` | ROS 机器人在线 | `RosRobot` + Rich TUI 实时帧率/动作时延表（`infer.py` 的 ROS 前身）。 |
| `scripts/value_server.py` | `ValueServer` | 与 `PolicyServer` 同协议的轻量版，对外提供 `value` 张量预测。 |
| `scripts/serve_policy.sh` | 启动 server | 支持 `CUDA_VISIBLE_DEVICES → port=9000+idx` 自动分配。 |
| `scripts/infer.sh` | 启动在线推理 | 包装 `tools/infer.py`，内含 SO101 默认摄像头/串口配置。 |
| `scripts/robot_client.sh` | 启动客户端 | 等价于 `python -m lerobot.async_inference.robot_client`，作为外部参照。 |
| `scripts/eval_policy.sh` | 评估 | `lerobot-record` + 指定 `--policy.path` 回放。 |
| `scripts/replay.sh` | 回放 | 调 `scripts/replay.py` 的 `replay_bot` 机器人类型。 |

**协议要点（与 `policy_client` 配套）**：

- Request：`{"__request_policy_name__": ...}` / `{"__request_policy_config__": ...}` / `{"observation": {...}}`
- Response：`{"action": Tensor[B,T,D]}` 或 `{"value": Tensor[B]}`
- 合并策略：`chunk_size_threshold` 阈值 + `aggregate_fn_name`（`weighted_average` / `latest_only`）

---

## 二、Value 模型 + ACP（优势条件策略）子系统

Pistar05/06 一条线：用 Value 模型对离线轨迹打分 → 计算 N-step Advantage → 训练 Advantage-Conditioned Policy。

| 路径 | 角色 | 关键能力 |
| --- | --- | --- |
| `scripts/train_value.py` | `value_train()` | Pistar06 value 模型训练入口；引入 `raw-batch hook` 断言。 |
| `scripts/train_value.sh` | 启动训练 | `--value.type=pistar06` + `lerobot_v3.0` 数据路径模板。 |
| `scripts/infer_value.py` | 离线批量打分 | 分布式 sampler、按 episode 计算 N-step advantage / 任务阈值、`complementary_info.{value,advantage,acp_indicator}` 写回 parquet。 |
| `scripts/infer_value.sh` | 启动批量推理 | 启用 `--acp.enable=true`，落盘到 `outputs/value_infer/`。 |
| `scripts/train_acp.py` | ACP 策略训练 | 在 `pi05` 基础上叠加 ACP hook（`acp_indicator` dropout 注入、value-conditioned）。 |
| `scripts/train_acp.sh` | 启动 ACP 训练 | 加载 `pi05_base` 预训练 + ACP 配置。 |
| `scripts/train_pistar.sh` | 占位 | 预留的 Pistar 训练脚本入口，目前为空。 |

**字段约定（与 dataset `complementary_info` 列对应）**：

- `value`：value 模型对当前状态的标量估计
- `advantage`：N-step advantage `R_t - V(s_t)`
- `acp_indicator`：0/1 标记是否参与 ACP 训练，受 `acp.indicator_dropout_prob` 控制

---

## 三、数据集 / 可视化 / 相机工具子系统

| 路径 | 角色 | 关键能力 |
| --- | --- | --- |
| `scripts/tools/ds_version_convert/` | 数据集版本迁移 | 覆盖 v1.6↔v2.0↔v2.1↔v3.0 的双向迁移脚本（commit `9c1fbbde`），子目录各自带 README。 |
| `scripts/convert_v21_to_v30.sh` | 一键 v2.1→v3.0 | 上述工具的便捷 shell。 |
| `scripts/tools/label_outcomes.py` | 批量标注 episode 结果 | `--all-outcome=success/failure` 或按 episode 列表标记，写入 metadata 的 `outcome` 字段（供 Value/ACP 使用）。 |
| `scripts/label_outcomes.sh` | 启动标签 | 默认对 `record_0429` 全量标 success（可改注释切到 failure 或精确列表）。 |
| `scripts/tools/dasetset_info.py` | 数据集摘要（文件名沿用项目历史拼写 `dasetset`） | 列出帧数、时长、字节数、视频/数据 feature 等；用于训练前自检。 |
| `scripts/tools/compare_actions.py` | 动作对比可视化 | 把不同策略（`pi05` vs `smolvla`）在同 episode 的 action / EEF 轨迹画到同一张图，算相似度。 |
| `scripts/tools/display_cameras.py` | 摄像头扫描预览 | `find_available_cameras()` + OpenCV 实时预览，用于调试 USB 摄像头编号。 |

---

## 四、相对 LeRobot 上游的差异总览

| 类别 | 自研 vs 上游 |
| --- | --- |
| 训练入口（policy） | **上游** `lerobot-train`，仅做配置壳。其中 `train_pi05.sh` 是项目内 pi05 训练的唯一标准入口，详见「0.1」。 |
| 数据采集 / 标定 / 编辑 | **上游** `lerobot-record` / `lerobot-calibrate` / `lerobot-edit-dataset` —— `record.sh` / `calibrate.sh` / `merge_datasets.sh` / `tag_dataset.sh` 仅为薄壳。 |
| 异步推理 | **自研** —— 上游未提供 ZMQ server / TUI dashboard / value server；本仓库从零实现并经历 6+ 轮重构（commit `0832d9c3 → ba8346f5`）。其中 `serve_policy.sh` 是项目内启 server 的唯一推荐入口，详见「0.2」。 |
| Value / ACP 训练 | **自研** —— `train_value.py` / `train_acp.py` / `infer_value.py` 均无上游对应。 |
| 数据集工具 | 部分**上游**（`ds_version_convert` 来自官方 PR `302/461/711/1412`，已搬迁到本目录）+ 部分**自研**（`label_outcomes` / `dasetset_info`（项目历史拼写）/ `compare_actions` / `display_cameras`）。 |

**训练入口对应的薄壳 shell**（全部包装 `lerobot-train`，无自研逻辑；pi05 的两个变体脚本名沿用项目历史拼写）：

```text
train_act.sh        train_smolvla.sh   train_policy.sh
train_pi05.sh       finetune_pi05.sh
```

---

## 五、调试入口速查

```bash
# ===== 重点入口（详见「〇」节）=====
# pi05 策略训练 → 产物 sorel/pi05-<dataset>
./scripts/train_pi05.sh <dataset>

# 启 Policy Server（端口随 CUDA_VISIBLE_DEVICES 自动）
./scripts/serve_policy.sh <policy_type> <ckpt>    # e.g. pi05 outputs/train_acp/record_0429/checkpoints/last

# ===== 自研工具 (待充分测试验证)=====
# Dashboard 批量管 server
python scripts/tools/policy_dashboard.py

# 在线推理（连 serve_policy.sh 起的 server）
./scripts/infer.sh "pick the tape and place it on the pad."

# Value 批量打分（喂 ACP 训练）
./scripts/infer_value.sh record_0429 outputs/value_train/record_0429/checkpoints/last

# 训练 ACP 策略（喂 serve_policy.sh 服务化）
./scripts/train_acp.sh record_0429 30000

# 数据集版本迁移
./scripts/convert_v21_to_v30.sh

# 标 episode 结果（喂 value 训练）
./scripts/label_outcomes.sh
```
