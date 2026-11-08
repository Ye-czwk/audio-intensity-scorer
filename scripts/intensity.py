#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""音频"激烈程度"(acoustic intensity)分析器。

只回答一个问题：这段音频有多激烈（能量密度/节奏密度/频谱活跃度），不给情绪正负、不给场景。

用法:
    python intensity.py <文件或目录> [--json] [--no-musicnn] [--window 8] [--windows 3]

输出: 0-100 总分 + 分档 + 各分项原始值与归一值。
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import soundfile as sf
import soxr
import torch

# MusicNN(MTT) 的 50 个标签顺序，来自 oriyonay/musicnn-pytorch 的 musicnn_torch.py: MTT_LABELS
MTT_LABELS = [
    "guitar", "classical", "slow", "techno", "strings", "drums", "electronic", "rock",
    "fast", "piano", "ambient", "beat", "violin", "vocal", "synth", "female", "indian",
    "opera", "male", "singing", "vocals", "no vocals", "harpsichord", "loud", "quiet",
    "flute", "woman", "male vocal", "no vocal", "pop", "soft", "sitar", "solo", "man",
    "classic", "choir", "voice", "new age", "dance", "male voice", "female vocal",
    "beats", "harp", "cello", "no voice", "weird", "country", "metal", "female voice", "choral",
]
LABEL_IDX = {name: i for i, name in enumerate(MTT_LABELS)}

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}

# ---------------------------------------------------------------- 归一化锚点
# 均为"经验锚点"，不是论文标定值：下限=几乎不激烈，上限=已经很激烈，超出即截断。
ANCHORS = {
    "tempo": (60.0, 160.0),          # BPM
    "onset_rate": (0.5, 9.0),        # 每秒起音数（普通流行 3-5，密集 16 分音符钢琴可达 8+）
    "rms_db": (-40.0, -8.0),         # 20*log10(RMS)；流媒体母带多在 -14 ~ -9 dB
    "dyn_db": (5.0, 25.0),           # 帧 RMS 的 p95-p5 动态范围(dB)
    "centroid": (500.0, 5000.0),     # Hz，对数刻度
    "rolloff": (1500.0, 9000.0),     # Hz，对数刻度
    "flux": (0.10, 0.60),            # 归一化正通量均值
}


def _lin(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _log(x, lo, hi):
    x = max(x, 1e-6)
    return float(np.clip((np.log10(x) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0.0, 1.0))


def read_audio(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def make_windows(y, sr, n_windows=3, win_sec=8.0):
    """取多个窗口而不是只取开头：试听片段/歌曲开头常常是安静引子（例如《Merry Christmas
    Mr. Lawrence》前 30 秒 RMS 只有 0.0288），只看开头会系统性低估。"""
    total = len(y) / sr
    win = min(win_sec, total)
    if total < win * 1.5:
        return [(0, len(y))]
    starts = []
    for frac in ([0.5] if n_windows == 1 else np.linspace(0.2, 0.8, n_windows)):
        s = int(max(0, min(total - win, total * frac - win / 2)) * sr)
        starts.append((s, s + int(win * sr)))
    return starts


def window_feats(y, sr):
    n_fft = 2048
    hop = 512
    win = np.hanning(n_fft)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop] * win
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)

    # 能量
    rms_frames = np.sqrt((frames ** 2).mean(axis=1))
    rms = float(rms_frames.mean())
    rms_db = 20 * np.log10(max(rms, 1e-6))
    dyn_db = float(20 * np.log10(max(np.percentile(rms_frames, 95), 1e-6) /
                                max(np.percentile(rms_frames, 5), 1e-6)))

    # 频谱
    centroid = float((mag * freqs).sum(1).mean() / (mag.sum(1).mean() + 1e-9))
    cum = np.cumsum(mag, axis=1)
    rolloff = float(np.mean([freqs[np.searchsorted(c, 0.85 * c[-1])] for c in cum]))
    norm_mag = mag / (mag.sum(1, keepdims=True) + 1e-9)
    flux = float(np.mean(np.maximum(np.diff(norm_mag, axis=0), 0).sum(1)))
    flatness = float(np.mean(np.exp(np.log(mag + 1e-6).mean(1)) / (mag.mean(1) + 1e-6)))

    # 节奏：用谱通量做起音包络（比 RMS 差分稳），并做峰值挑选，避免安静段落把噪声数成起音
    sp = np.log1p(mag * 100)
    onset_env = np.maximum(np.diff(sp, axis=0), 0).mean(axis=1)
    fps = sr / hop
    dur = len(y) / sr
    p95 = float(np.percentile(onset_env, 95)) if onset_env.size else 0.0
    thr = max(0.15 * p95, float(onset_env.mean() + 0.5 * onset_env.std()))
    min_gap = max(1, int(fps * 0.05))  # 峰值最小间隔 50ms
    peaks, last = [], -min_gap
    for i in range(1, len(onset_env) - 1):
        if (onset_env[i] >= thr and onset_env[i] == onset_env[i - 1:i + 2].max()
                and i - last >= min_gap):
            peaks.append(i)
            last = i
    onset_rate = len(peaks) / dur

    # 节拍：包络自相关；八度折叠到 80-160 BPM，并给出周期性置信度
    e = onset_env - onset_env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    lo_lag, hi_lag = int(fps * 60 / 200), int(fps * 60 / 50)
    tempo, periodicity = 0.0, 0.0
    if hi_lag < len(ac) and ac[0] > 1e-9:
        lag = lo_lag + int(np.argmax(ac[lo_lag:hi_lag]))
        tempo = 60.0 * fps / lag
        periodicity = float(np.clip(ac[lag] / ac[0], 0.0, 1.0))
        while tempo < 80 and tempo > 0:
            tempo *= 2
        while tempo > 160:
            tempo /= 2

    return {"tempo_bpm": tempo, "periodicity": periodicity, "onset_rate": onset_rate,
            "rms": rms, "rms_db": rms_db, "dyn_db": dyn_db,
            "centroid_hz": centroid, "rolloff_hz": rolloff,
            "flux": flux, "flatness": flatness}


def musicnn_probs(path, cache):
    """返回 MTT 50 标签概率。复刻 musicnn.py 的预处理：soundfile 读入 -> soxr 重采样 -> mel(96) ->
    3 秒窗口 -> sigmoid。

    注意：这里没有用 librosa。本机实测 librosa 的 load/resample/melspectrogram 都会卡死（numba 路径
    与 torch 共存时不可用），所以 mel 用自带实现（Slaney 刻度 + Slaney 面积归一），与
    librosa.feature.melspectrogram(sr=16000, n_fft=512, hop=256, n_mels=96, power=2) 等价。
    """
    if "model" not in cache:
        from transformers import AutoModel
        cache["model"] = AutoModel.from_pretrained("oriyonay/musicnn-pytorch", trust_remote_code=True).eval()
    model = cache["model"]

    y, sr = read_audio(path)
    if sr != 16000:
        y = soxr.resample(y, sr, 16000)
    mel = _melspectrogram(y, 16000, 512, 256, 96).T
    mel = np.log10(10000 * mel.astype(np.float32) + 1)
    n_frames = 187
    last = mel.shape[0] - n_frames + 1
    if last <= 0:
        patch = np.zeros((n_frames, 96), dtype=np.float32)
        patch[: mel.shape[0], :] = mel
        batches = [patch]
    else:
        batches = [mel[t:t + n_frames, :] for t in range(0, last, n_frames)]

    probs = []
    with torch.no_grad():
        for b in batches:
            logits, _, _ = model(torch.from_numpy(np.stack([b])))
            probs.append(torch.sigmoid(logits).squeeze(0).numpy())
    return dict(zip(MTT_LABELS, np.mean(probs, axis=0).tolist()))


def _hz_to_mel(hz):
    """Slaney 刻度（librosa 默认 htk=False）。标量与数组都支持。"""
    hz = np.asarray(hz, dtype=float)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.0) / 27.0
    return np.where(hz < min_log_hz, hz / f_sp,
                    min_log_mel + np.log(np.maximum(hz, 1e-12) / min_log_hz) / logstep)


def _mel_to_hz(mels):
    mels = np.asarray(mels, dtype=float)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.0) / 27.0
    return np.where(mels < min_log_mel, f_sp * mels,
                    min_log_hz * np.exp(logstep * (mels - min_log_mel)))


def _mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = sr / 2.0 if fmax is None else fmax
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - freqs) / max(hi - ce, 1e-9)
        fb[i] = np.maximum(0, np.minimum(up, down))
        fb[i] *= 2.0 / max(hi - lo, 1e-9)  # Slaney 面积归一
    return fb


def _melspectrogram(y, sr, n_fft, hop, n_mels):
    win = np.hanning(n_fft + 1)[:-1]  # 周期 hann，与 librosa 一致
    y = np.pad(y, (n_fft // 2, n_fft // 2), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop] * win
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return _mel_filterbank(sr, n_fft, n_mels) @ power.T


def score_file(path, use_musicnn=True, n_windows=3, win_sec=8.0, cache=None):
    cache = cache if cache is not None else {}
    y, sr = read_audio(path)
    ws = make_windows(y, sr, n_windows, win_sec)
    per = [window_feats(y[a:b], sr) for a, b in ws]
    agg = {k: float(np.median([p[k] for p in per])) for k in per[0]}

    n = {
        "tempo": _lin(agg["tempo_bpm"], *ANCHORS["tempo"]),
        "periodicity": _lin(agg["periodicity"], 0.0, 0.6),
        "onset_rate": _lin(agg["onset_rate"], *ANCHORS["onset_rate"]),
        "rms_db": _lin(agg["rms_db"], *ANCHORS["rms_db"]),
        "dyn_db": _lin(agg["dyn_db"], *ANCHORS["dyn_db"]),
        "centroid": _log(agg["centroid_hz"], *ANCHORS["centroid"]),
        "rolloff": _log(agg["rolloff_hz"], *ANCHORS["rolloff"]),
        "flux": _lin(agg["flux"], *ANCHORS["flux"]),
    }
    # 节拍分要考虑"自相关峰值是否可信"：周期置信度低时把 tempo 分打折
    tempo_eff = n["tempo"] * (0.4 + 0.6 * n["periodicity"])

    # 节奏分 = 节拍(0.4) + 起音密度(0.6)；再乘一个"响度闸门"：
    # 极安静的音频即使起音很密，也不该被判成激烈（本机实测《Merry Christmas》前 30 秒 RMS≈0.0288）
    gate = float(np.clip((agg["rms_db"] + 40.0) / 28.0, 0.0, 1.0))
    rhythm = (0.40 * tempo_eff + 0.60 * n["onset_rate"]) * (0.25 + 0.75 * gate)
    energy = 0.65 * n["rms_db"] + 0.35 * n["dyn_db"]
    spectrum = 0.50 * n["centroid"] + 0.30 * n["rolloff"] + 0.20 * n["flux"]

    model_tags = None
    model_pos = None
    if use_musicnn:
        probs = musicnn_probs(path, cache)
        keys = ["fast", "loud", "beat", "dance"]
        model_pos = float(np.mean([probs[k] for k in keys]))
        # MTT 是 50 类多标签 sigmoid，输出绝对值普遍偏低（本批 10 段实测 pos 最高 0.203），
        # 若把原概率直接当 0-1 用，模型项在总分里近乎失效（只值 0-1.5 分）。因此按参考上限
        # 0.25 线性映射到 0-1。0.25 是经验值，未经外部数据标定。
        model_score = float(np.clip(model_pos / 0.25, 0.0, 1.0))
        model_tags = {k: round(probs[k], 3) for k in keys + ["slow", "quiet", "soft"]}
    else:
        model_score = None

    if model_score is None:
        w = {"rhythm": 0.35, "energy": 0.35, "spectrum": 0.30}
        total = w["rhythm"] * rhythm + w["energy"] * energy + w["spectrum"] * spectrum
    else:
        w = {"rhythm": 0.30, "energy": 0.30, "spectrum": 0.25, "model": 0.15}
        total = (w["rhythm"] * rhythm + w["energy"] * energy +
                 w["spectrum"] * spectrum + w["model"] * model_score)

    return {
        "file": os.path.basename(path),
        "duration_sec": round(len(y) / sr, 2),
        "windows": len(ws),
        "raw": {k: round(v, 2) for k, v in agg.items()},
        "normalized": {k: round(v, 3) for k, v in n.items()},
        "subscores": {"rhythm": round(rhythm, 3), "energy": round(energy, 3),
                      "spectrum": round(spectrum, 3),
                      "musicnn": None if model_score is None else round(model_score, 3)},
        "musicnn_pos_mean": None if model_pos is None else round(model_pos, 3),
        "weights": w,
        "musicnn_tags": model_tags,
        "score": round(total * 100, 1),
        "band": band_of(total * 100),
    }


def band_of(s):
    if s < 35:
        return "低"
    if s < 55:
        return "中低"
    if s < 70:
        return "中"
    if s < 85:
        return "高"
    return "极高"


def main():
    ap = argparse.ArgumentParser(description="音频激烈程度分析")
    ap.add_argument("path", help="音频文件或目录")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    ap.add_argument("--no-musicnn", action="store_true", help="跳过 MusicNN（更快，只用声学特征）")
    ap.add_argument("--window", type=float, default=8.0, help="分析窗口长度(秒)，默认 8")
    ap.add_argument("--windows", type=int, default=3, help="窗口数量，默认 3（取 20%%/50%%/80%% 位置）")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(os.path.join(args.path, f) for f in os.listdir(args.path)
                       if os.path.splitext(f)[1].lower() in AUDIO_EXT)
    else:
        files = [args.path]

    cache = {}
    results = []
    for f in files:
        try:
            results.append(score_file(f, not args.no_musicnn, args.windows, args.window, cache))
        except Exception as e:
            results.append({"file": os.path.basename(f), "error": f"{type(e).__name__}: {e}"})

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
        return

    hdr = f"{'文件':22s} {'BPM':>6s} {'起音/s':>7s} {'RMS dB':>7s} {'动态 dB':>7s} {'质心Hz':>7s} {'通量':>5s} {'节奏':>5s} {'能量':>5s} {'频谱':>5s} {'模型':>5s} {'总分':>6s}  档"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['file']:22s} ERROR {r['error']}")
            continue
        raw, sub, m = r["raw"], r["subscores"], r["musicnn_tags"]
        if m:
            ms = sum(m[k] for k in ("fast", "loud", "beat", "dance")) / 4
            model_col = f"{ms:5.2f}"
        else:
            model_col = "    -"
        print(f"{r['file']:22s} {raw['tempo_bpm']:6.1f} {raw['onset_rate']:7.2f} "
              f"{raw['rms_db']:7.1f} {raw['dyn_db']:7.1f} {raw['centroid_hz']:7.0f} "
              f"{raw['flux']:5.2f} {sub['rhythm']:5.2f} {sub['energy']:5.2f} "
              f"{sub['spectrum']:5.2f} {model_col} "
              f"{r['score']:6.1f}  {r['band']}")
    print("\n注：分数是声学激烈度，不等于情绪好坏，也不等于适合运动。")


if __name__ == "__main__":
    main()
