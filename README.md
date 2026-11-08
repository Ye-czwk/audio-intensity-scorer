# audio-intensity-scorer

给离线音频打一个「激烈程度」分数（0–100）并分档，同时输出各分项的实测值。

## 这是什么、不是什么

**它是**：一个声学测量工具。读一个本地音频文件（或一个目录），按声学上的能量、节奏密度、频谱活跃度算出一个综合分，落在 0–100，并按阈值分档（低 / 中低 / 中 / 高 / 极高）；除总分外，它把每个原始测量值也打印出来。

输出的内容具体是：

- 分项分：节奏分、能量分、频谱分、模型分（0–1），以及总分（0–100）与分档
- 原始值：BPM、起音密度（起音/秒）、RMS、动态范围、频谱质心、rolloff、谱通量
- 模型原始输出：MusicNN 的 fast / loud / beat / dance / slow / quiet / soft 标签概率

**它不是**：

- 不是音频标签器（上面列的标签概率是模型的中间产物，不是本工具的产品形态）
- 不是曲风分类器
- 不是情绪识别工具（不判情绪正负）
- 不判有没有人声
- 不判是否适合运动、助眠、通勤等场景

换句话说：分数只说明「这段录音有多响、多密、多亮」，不说明「这首歌是什么、好不好听、适合干什么」。

## 已知局限（先读这段，再决定要不要用）

1. **准确率没有保证。** 在这批 10 段样本上，排序方向（「A 比 B 更激烈」）与普通听众的一致率约 **7–8 成**；**分档的绝对正确率更低，约 5–6 成**。这两个数字是作者的主观估计，不是标准化测评结果，也没有任何人工标注数据做回归或 ROC 标定。本仓库不承诺准确率。
2. **三个已知的失手场景：**
   - **独奏钢琴 / 古典 / 无鼓编制**：频谱暗、模型里 beat/dance 接近零，会被系统性压低。《Rush E》速度极快、音符极密，实测只有 44.4 分（低于《晴天》）。
   - **试听片段恰好是压缩副歌的抒情流行**：这段音频响度高、动态范围小，会被系统性抬高。《晴天》实测 50.6 分，排全场第 3。
   - **宽动态范围的慢歌**：动态范围在公式里是正权重，宽动态既可能是「有爆发」，也可能是「大段安静」。《漠河舞厅》的动态范围 30.0 dB 被截顶，能量分 0.75 是全表第二高。这是当前公式已知的设计问题，未为了某首歌好看去调参。
3. **总分权重、归一化锚点、分档阈值全部是经验值，未经标定。** 它们是按声学常识加这 10 段音频的分布定的。**建议优先看分项值和原始测量值，而不是只看总分**；如果你有带人工打分的音频集，应该重新标定这些数字。

## 安装

```powershell
py -3.11 -m pip install numpy soundfile soxr torch "transformers==4.46.3"
```

已实测可用的版本组合：numpy 2.4.6 / soundfile 0.14.0 / soxr 1.1.0 / torch 2.15.0+cpu / transformers 4.46.3（Python 3.11）。

依赖上的硬要求：

- **`transformers` 必须锁 `4.46.3`**。5.x 加载 MusicNN 的自定义模型会因 `all_tied_weights_keys` 报错失败。
- **不要引入 `librosa`**。在部分环境（Windows + torch 共存）里 `librosa.load` / `resample` / `melspectrogram` 会卡死且无日志。本脚本完全不 import librosa：I/O 用 `soundfile`，重采样用 `soxr`，起音与 tempo 用 numpy 自相关，mel 用自实现的 Slaney 滤波器组。
- **不要把 Essentia 列为依赖**。它在 Windows 上装不上（PyPI 无 Windows wheel）。

模型权重来自 HuggingFace 的 [`oriyonay/musicnn-pytorch`](https://huggingface.co/oriyonay/musicnn-pytorch)，约 3 MB，**首次运行会自动下载**（脚本内置 `HF_ENDPOINT=https://hf-mirror.com`）。

## 用法

```powershell
# 单个文件
python scripts/intensity.py "D:\music\song.mp3"

# 整个目录（按文件名排序批量跑）
python scripts/intensity.py "D:\music\clips"

# 输出 JSON，便于二次处理
python scripts/intensity.py "D:\music\clips" --json

# 跳过 MusicNN（快很多，只用声学特征，权重自动重新归一）
python scripts/intensity.py "D:\music\clips" --no-musicnn

# 调整分析窗口
python scripts/intensity.py "D:\music\clips" --window 8 --windows 3
```

时间成本（本机实测）：单个文件端到端约 80 秒，大头是导入 torch/transformers 并加载模型；同一进程里每多一个 30 秒文件约 +10 秒。批量时模型只加载一次。

终端输出示例列头：

```
文件  BPM  起音/s  RMS dB  动态 dB  质心Hz  通量  节奏  能量  频谱  模型  总分  档
```

## 怎么算的（简述）

- **分析窗口**：默认取 3 个 8 秒窗口（文件 20% / 50% / 80% 位置），各窗口算特征后**取中位数**。只看开头会把「极安静引子」和「录坏了」混为一谈。
- **归一化**：BPM 60–160、起音 0.5–9.0/s、RMS −40 到 −8 dB、动态 5–25 dB、质心 500–5000 Hz（对数）、rolloff 1500–9000 Hz（对数）、通量 0.10–0.60。全部是经验锚点，超界截断。
- **分项**：
  - 节奏分 = (0.40 × tempo 项 + 0.60 × 起音密度项) × 响度闸门；tempo 项还会按自相关周期置信度打折。
  - 能量分 = 0.65 × RMS 项 + 0.35 × 动态范围项。
  - 频谱分 = 0.50 × 质心项 + 0.30 × rolloff 项 + 0.20 × 通量项。
  - 模型分 = mean(P(fast), P(loud), P(beat), P(dance)) / 0.25，截断到 0–1（MTT 是多标签 sigmoid，原始概率绝对值普遍很低，0.25 是经验上限）。
- **权重**：有模型分时 节奏 0.30 / 能量 0.30 / 频谱 0.25 / 模型 0.15；`--no-musicnn` 时 0.35 / 0.35 / 0.30。tempo 在总分里的实际贡献通常只有 5–9%，是刻意压低的，避免把「快 BPM 的抒情慢歌」误判成激烈。
- **分档**：`<35 低`、`<55 中低`、`<70 中`、`<85 高`、`≥85 极高`（经验阈值，未经标定）。

## 实测记录

10 段 iTunes 30 秒试听，解码为 48 kHz wav，全部按默认参数（3 窗口 × 8 秒，取中位数）跑一遍。这是工具的第一次验证，不是基准测评；样本量很小，且三首失手案例都在这批里。

| 曲目 | 总分 | 分档 |
|---|---|---|
| All Falls Down | 67.2 | 中 |
| Take a Walk | 56.9 | 中 |
| 晴天 | 50.6 | 中低 |
| 形容 | 49.3 | 中低 |
| 漠河舞厅 | 48.4 | 中低 |
| Rush E | 44.4 | 中低 |
| 童话镇 | 43.1 | 中低 |
| 孤勇者 | 42.6 | 中低 |
| River Flows in You | 25.1 | 低 |
| Merry Christmas Mr. Lawrence | 18.7 | 低 |

成立的部分：All Falls Down 比 Merry Christmas 高 48.5 分（约 3.6 倍）；两首纯钢琴垫底，两首电音/摇滚流行居首。不成立的部分见上文「三个已知的失手场景」。

完整原始数据（各特征值、分项分、MusicNN 概率）来自测试时的 `results_full.json`，该文件为中间产物且对应音频有版权，**不随仓库发布**。

## 目录

```
audio-intensity-repo/
├── README.md             # 主文档，用法以本文件为准
├── SKILL.md              # 附带的 Trae skill 格式封装，内容与 README 有重叠
├── LICENSE               # MIT
├── requirements.txt
└── scripts/
    └── intensity.py      # 全部实现，单文件
```

`SKILL.md` 只是把它套成 Trae skill 格式的附带封装，**主用法以本 README 为准**。

## License

MIT，见 [LICENSE](LICENSE)。
