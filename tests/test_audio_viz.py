"""音频可视化分析层：Web API 与 matplotlib 窗口共用的返回结构。

这些测试不检查绘图本身（Canvas 与 matplotlib 只消费数据），而是锁定前端依赖的
契约：数组必须下采样到可传输的规模、CSI 与趋势必须对齐、录音不合格时返回空图
而不是抛出异常。
"""
from pathlib import Path

import numpy as np
import pytest

from core.audio_viz import (
    FRAME_POINTS,
    SPEC_FREQ_POINTS,
    SPEC_TIME_POINTS,
    SPECTRUM_POINTS,
    analyze_audio,
    analyze_bytes,
)
from core.replay import render_wav, write_wav


@pytest.fixture(scope="module")
def rendered_wav(tmp_path_factory):
    """渲染一段包含 Kasami 探测音的座舱录音，供多个用例复用。"""
    path = tmp_path_factory.mktemp("viz") / "sleeping.wav"
    render_wav(path, "sleeping", duration_s=24.0, snr_db=40.0)
    return path


def test_analyze_audio_returns_plot_ready_arrays(rendered_wav):
    data = analyze_audio(rendered_wav)

    assert data["qc"]["status"] in {"PASS", "WARN"}, data["qc"]
    assert data["fs"] == 48000
    assert 20.0 <= data["duration_s"] <= 30.0

    sg = data["spectrogram"]
    assert 0 < len(sg["t"]) <= SPEC_TIME_POINTS
    assert 0 < len(sg["f_hz"]) <= SPEC_FREQ_POINTS
    assert len(sg["db"]) == len(sg["f_hz"])
    assert all(len(row) == len(sg["t"]) for row in sg["db"])
    codes = [v for row in sg["db"] for v in row]
    assert 0 <= min(codes) and max(codes) <= 255
    assert sg["f_max_hz"] <= 20000
    assert sg["range_db"] > 0

    spec = data["spectrum"]
    assert 0 < len(spec["f_hz"]) <= SPECTRUM_POINTS
    assert len(spec["f_hz"]) == len(spec["db"])

    csi = data["csi"]
    assert 0 < len(csi["amp"]) <= FRAME_POINTS
    rows = len(csi["amp"][0])
    assert rows == data["band"]["n_carriers"]
    assert len(csi["carrier_hz"]) == rows
    assert len(csi["t"]) == len(csi["amp"])

    trends = data["trends"]
    assert len(trends["t"]) == len(trends["motion_ratio"])
    assert len(trends["t"]) == len(trends["coherence"]) == len(trends["bpm"])
    assert trends["threshold"] is not None

    assert data["summary"]["frames"] > 0
    names = {item["name"] for item in data["params"]}
    assert {"格式", "采样率", "电平", "帧同步", "存在判定", "运动相干性 κ"} <= names
    assert all(item["status"] in {"PASS", "WARN", "FAIL", "INFO"}
               for item in data["params"])


def test_max_seconds_truncates_analysis_but_keeps_full_duration(rendered_wav):
    """选择“前 N 秒”时只分析这一段，但时长质检仍按完整文件判定。"""
    data = analyze_audio(rendered_wav, max_seconds=10.0)

    assert data["duration_s"] <= 10.5
    assert data["file_duration_s"] >= 20.0
    assert len(data["csi"]["amp"]) < 600
    assert any(item["name"] == "分析长度" for item in data["params"])


def test_analyze_bytes_cleans_up_its_temp_file(rendered_wav):
    """上传路径不应把临时 WAV 留在磁盘上。"""
    data = analyze_bytes(rendered_wav.read_bytes())
    assert data["summary"]["frames"] > 0
    assert not Path(data["path"]).exists()


def test_analyze_without_probe_degrades_instead_of_raising(tmp_path):
    """录音里没有探测音时，接口返回空 CSI 与质检结论，而不是 500。"""
    rng = np.random.default_rng(0)
    path = write_wav(tmp_path / "noise.wav", rng.standard_normal(48000 * 3) * 0.1)

    data = analyze_audio(path)

    assert "no probe repetition" in data["error"]
    assert data["csi"] == {"t": [], "amp": []}
    assert data["trends"] == {}
    assert data["events"] == []
    assert data["summary"]["detected"] is False
    # 语谱图与频谱仍然可用：前端能显示“录到了什么”，只是无法恢复 CSI。
    assert data["spectrogram"]["t"]
    assert data["spectrum"]["f_hz"]
    assert all(item["status"] in {"PASS", "WARN", "FAIL", "INFO"}
               for item in data["params"])
