import numpy as np

from mockingbird.audio.loopback import _LinearResampler


def _assert_continuous_sine(src_rate: int, dst_rate: int) -> None:
    r = _LinearResampler(src_rate, dst_rate)
    n = src_rate * 3
    x = np.sin(2 * np.pi * 440 * np.arange(n) / src_rate).astype(np.float32)
    out = []
    for i in range(0, n, src_rate // 10):
        out.append(r.process(x[i : i + src_rate // 10]))
    y = np.concatenate(out)
    assert abs(len(y) - dst_rate * 3) <= 2
    # A 440 Hz sine at the dst rate has a small max slope; a discontinuity at a
    # block boundary would show up as an implausibly large jump.
    d = np.abs(np.diff(y))
    assert float(np.max(d)) < 0.5


def test_downsample_48k_to_16k():
    _assert_continuous_sine(48000, 16000)


def test_downsample_44k_to_16k():
    _assert_continuous_sine(44100, 16000)


def test_resample_length_ratio():
    r = _LinearResampler(48000, 16000)
    out = []
    for _ in range(10):
        out.append(r.process(np.zeros(4800, dtype=np.float32)))
    total = sum(len(b) for b in out)
    assert total == 16000  # 10 * 4800 source samples == 10 * 1600 output samples


def test_empty_block():
    r = _LinearResampler(48000, 16000)
    assert r.process(np.zeros(0, dtype=np.float32)).shape == (0,)
