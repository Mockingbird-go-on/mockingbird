import numpy as np

from mockingbird.audio.loopback import _BandLimitedResampler, _LinearResampler, _LoopbackAgc


def _run_blocks(resampler, x, src_rate, block_s=0.1):
    block = int(src_rate * block_s)
    out = []
    for i in range(0, len(x), block):
        out.append(resampler.process(x[i : i + block]))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _assert_continuous_sine(src_rate: int, dst_rate: int) -> None:
    r = _LinearResampler(src_rate, dst_rate)
    n = src_rate * 3
    x = np.sin(2 * np.pi * 440 * np.arange(n) / src_rate).astype(np.float32)
    y = _run_blocks(r, x, src_rate)
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


# --- BandLimitedResampler -------------------------------------------------


def test_bandlimited_sine_passes():
    """A 1 kHz tone must survive 48k→16k with near-full amplitude."""
    src = 48000
    n = src * 2
    t = np.arange(n) / src
    x = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    r = _BandLimitedResampler(src, 16000)
    y = _run_blocks(r, x, src)
    # Length: 2 s minus the filter warm-up (K-1 input samples)
    assert abs(len(y) - 2 * 16000) <= 160
    mid = y[len(y) // 4 : 3 * len(y) // 4]
    assert float(np.max(np.abs(mid))) > 0.35


def test_bandlimited_alias_attenuated():
    """A 12 kHz tone (above the 8 kHz output Nyquist) must be suppressed."""
    src = 48000
    n = src * 2
    t = np.arange(n) / src
    x = (0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
    r = _BandLimitedResampler(src, 16000)
    y = _run_blocks(r, x, src)
    mid = y[len(y) // 4 : 3 * len(y) // 4]
    ref = _BandLimitedResampler(src, 16000)
    t2 = np.arange(src * 2) / src
    ref_y = _run_blocks(ref, (0.5 * np.sin(2 * np.pi * 1000 * t2)).astype(np.float32), src)
    ref_mid = ref_y[len(ref_y) // 4 : 3 * len(ref_y) // 4]
    alias_rms = float(np.sqrt(np.mean(np.square(mid))))
    ref_rms = float(np.sqrt(np.mean(np.square(ref_mid))))
    assert ref_rms > 0.3
    # At least 30 dB (×0.03) suppression of the aliasing tone
    assert alias_rms < ref_rms * 0.03, f"alias rms={alias_rms:.4f} ref={ref_rms:.4f}"


def test_bandlimited_linear_alias_not_attenuated():
    """Sanity: the old linear resampler lets the 12 kHz tone alias through,
    which is exactly why the band-limited one exists."""
    src = 48000
    n = src * 2
    t = np.arange(n) / src
    x = (0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
    y = _run_blocks(_LinearResampler(src, 16000), x, src)
    mid = y[len(y) // 4 : 3 * len(y) // 4]
    assert float(np.sqrt(np.mean(np.square(mid)))) > 0.1


def test_bandlimited_no_samples_lost_or_duplicated():
    """Long-run output length matches dst_rate exactly (warm-up accounted)."""
    src = 48000
    x = np.zeros(src * 5, dtype=np.float32)
    y = _run_blocks(_BandLimitedResampler(src, 16000), x, src)
    assert abs(len(y) - 5 * 16000) <= 160


def test_bandlimited_44k_length():
    src = 44100
    x = np.zeros(src * 3, dtype=np.float32)
    y = _run_blocks(_BandLimitedResampler(src, 16000), x, src)
    assert abs(len(y) - 3 * 16000) <= 160


def test_bandlimited_identity():
    r = _BandLimitedResampler(16000, 16000)
    x = np.ones(100, dtype=np.float32)
    out = r.process(x)
    assert np.array_equal(out, x)


def test_bandlimited_empty_block():
    r = _BandLimitedResampler(48000, 16000)
    assert r.process(np.zeros(0, dtype=np.float32)).shape == (0,)


def test_bandlimited_small_blocks_no_gaps():
    """Feed 10 ms blocks: every call after the first must produce output
    (no buffer deadlock), and amplitude of a 1 kHz tone is preserved."""
    src = 48000
    n = src * 2
    t = np.arange(n) / src
    x = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    r = _BandLimitedResampler(src, 16000)
    out = []
    for i in range(0, n, src // 100):
        out.append(r.process(x[i : i + src // 100]))
    y = np.concatenate(out)
    assert abs(len(y) - 2 * 16000) <= 160
    mid = y[len(y) // 4 : 3 * len(y) // 4]
    assert float(np.max(np.abs(mid))) > 0.35


# --- Loopback AGC ----------------------------------------------------------


def test_agc_amplifies_quiet_signal():
    agc = _LoopbackAgc(target_rms=0.07)
    x = np.full(16000, 0.02, dtype=np.float32)  # quiet remote caller
    outs = [agc.process(x.copy()) for _ in range(200)]  # 20 s
    y = np.concatenate(outs)
    steady = y[-16000:]
    rms = float(np.sqrt(np.mean(np.square(steady))))
    assert rms > 0.05


def test_agc_does_not_amplify_silence():
    agc = _LoopbackAgc()
    silence = np.zeros(16000, dtype=np.float32)
    y = agc.process(silence.copy())
    assert float(np.max(np.abs(y))) == 0.0


def test_agc_loud_signal_not_attenuated():
    agc = _LoopbackAgc()
    x = np.full(16000, 0.5, dtype=np.float32)
    y = agc.process(x.copy())
    assert float(np.max(np.abs(y))) > 0.45


def test_agc_never_clips():
    agc = _LoopbackAgc()
    # Alternating very quiet / loud blocks — gain pump stress test
    quiet = np.full(1600, 0.01, dtype=np.float32)
    loud = np.full(1600, 0.9, dtype=np.float32)
    for _ in range(50):
        agc.process(quiet.copy())
        y = agc.process(loud.copy())
        assert float(np.max(np.abs(y))) <= 0.98
