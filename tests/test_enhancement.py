import numpy as np
import pytest

from audiolab.enhancement import (MCRA, NoiseReducer, OracleWiener, SPPNoise, Wola, frame_sizes,
                                  process_signal, si_sdr, sqrt_hann)


@pytest.mark.parametrize("n_fft, overlap", [(64, 2), (128, 2), (512, 2), (64, 4), (512, 4)])
def test_perfect_reconstruction(n_fft, overlap):
    x = np.random.default_rng(0).standard_normal(5000)
    y = process_signal(x, Wola(n_fft, n_fft // overlap))
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_hann_squared_adds_up_to_a_constant():
    w2 = sqrt_hann(256) ** 2
    total = w2[:128] + w2[128:]               # two copies shifted by half a frame
    np.testing.assert_allclose(total, 1.0, atol=1e-12)


def test_delay_is_frame_minus_hop():
    wola = Wola(128, 32)
    impulse = np.zeros(32 * 20)
    impulse[0] = 1
    out = np.concatenate([wola.process(impulse[i:i + 32]) for i in range(0, len(impulse), 32)])
    assert np.argmax(np.abs(out)) == 128 - 32 == wola.delay


def test_a_constant_gain_scales_the_output():
    x = np.random.default_rng(1).standard_normal(3000)
    y = process_signal(x, Wola(256, 128), lambda X: 0.5 * X)
    np.testing.assert_allclose(y, 0.5 * x, atol=1e-10)


def test_zeroing_high_bins_is_a_low_pass():
    sr, n_fft = 16000, 256
    t = np.arange(sr) / sr
    low, high = np.sin(2 * np.pi * 500 * t), np.sin(2 * np.pi * 6000 * t)

    def keep_below_2k(X):
        X = X.copy()
        X[int(2000 * n_fft / sr):] = 0
        return X

    y = process_signal(low + high, Wola(n_fft, n_fft // 2), keep_below_2k)
    middle = slice(1000, -1000)               # away from the start-up and the end
    assert np.std(y[middle] - low[middle]) < 1e-3


def test_wrong_block_size_is_rejected():
    with pytest.raises(ValueError):
        Wola(128, 64).process(np.zeros(32))


def test_frame_sizes():
    assert frame_sizes(32, 16000) == (512, 256)
    assert frame_sizes(4, 16000, overlap=4) == (64, 16)


def _frames_of(x, n_fft, hop):
    """|X|^2 of every frame that a Wola would see."""
    rec = []
    process_signal(x, Wola(n_fft, hop), lambda X: rec.append(np.abs(X) ** 2) or X)
    return np.array(rec)


@pytest.mark.parametrize("Estimator", [MCRA, SPPNoise])
def test_noise_estimate_converges_on_white_noise(Estimator):
    sr, n_fft, hop = 16000, 256, 128
    x = np.random.default_rng(2).standard_normal(sr * 4)
    est = Estimator(n_fft // 2 + 1, hop / sr)
    for power in _frames_of(x, n_fft, hop):
        noise = est.update(power)
    # E|X|^2 per bin = sum(window^2) * sigma^2 for white noise; check the average over bins
    expected = np.sum(sqrt_hann(n_fft) ** 2)
    assert abs(10 * np.log10(np.mean(noise[5:-5]) / expected)) < 1.5


def _bursts(sr, seconds, on=0.3, off=0.3, f=1000):
    """A tone switched on and off like syllables: on seconds of tone, off seconds of silence."""
    t = np.arange(int(seconds * sr)) / sr
    gate = (t % (on + off)) < on
    return np.sin(2 * np.pi * f * t) * gate, gate


def test_spp_does_not_take_syllables_for_noise():
    sr, n_fft, hop = 16000, 256, 128
    noise = 0.1 * np.random.default_rng(3).standard_normal(sr * 4)
    tone, _ = _bursts(sr, 3)
    k = int(1000 * n_fft / sr)                            # the bin of the tone

    def track(x):
        est = SPPNoise(n_fft // 2 + 1, hop / sr)
        return np.array([est.update(p)[k] for p in _frames_of(x, n_fft, hop)])

    with_tone = noise.copy()
    with_tone[sr:] += tone                                # 27 dB above the noise in its bin
    # Same noise with and without the syllables: the estimate must barely notice them
    ratio = np.mean(track(with_tone)[sr // hop:]) / np.mean(track(noise)[sr // hop:])
    assert ratio < 1.12                                   # < 0.5 dB


def test_mcra_absorbs_the_onset_of_a_sound():
    # The known weakness of MCRA: its speech detector looks at a smoothed power that reacts a few
    # frames late, and meanwhile the noise estimate takes in the start of the sound
    sr, n_fft, hop = 16000, 256, 128
    x = 0.1 * np.random.default_rng(3).standard_normal(sr * 2)
    x[sr:] += np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)
    est = MCRA(n_fft // 2 + 1, hop / sr)
    k = int(1000 * n_fft / sr)
    track = np.array([est.update(p)[k] for p in _frames_of(x, n_fft, hop)])
    assert track[sr // hop + 2] > 10 * track[sr // hop - 1]   # +10 dB within 3 frames


def test_spp_tracks_a_noise_level_step():
    sr, n_fft, hop = 16000, 256, 128
    rng = np.random.default_rng(4)
    x = np.concatenate([0.01 * rng.standard_normal(2 * sr), 0.1 * rng.standard_normal(3 * sr)])
    est = SPPNoise(n_fft // 2 + 1, hop / sr)
    noise = [np.mean(est.update(p)) for p in _frames_of(x, n_fft, hop)]
    expected = 0.1 ** 2 * np.sum(sqrt_hann(n_fft) ** 2)   # +20 dB after the step
    assert abs(10 * np.log10(noise[-1] / expected)) < 2


def test_reducer_attenuates_noise_to_the_gain_floor():
    sr, n_fft, hop = 16000, 256, 128
    x = np.random.default_rng(5).standard_normal(sr * 4)
    reducer = NoiseReducer(SPPNoise(n_fft // 2 + 1, hop / sr), hop / sr, gain_floor_db=-15)
    y = process_signal(x, Wola(n_fft, hop), reducer)
    late = slice(sr, None)                                # once the noise estimate has settled
    attenuation = 10 * np.log10(np.mean(y[late] ** 2) / np.mean(x[late] ** 2))
    assert -17 < attenuation < -12


@pytest.mark.parametrize("rule", ["subtraction", "wiener", "wiener_dd", "wiener_tsnr"])
def test_reducer_keeps_strong_syllables(rule):
    sr, n_fft, hop = 16000, 256, 128
    tone, gate = _bursts(sr, 3)
    tone[:sr] = 0                                         # 1 s of noise only first
    x = tone + 0.01 * np.random.default_rng(6).standard_normal(len(tone))
    reducer = NoiseReducer(SPPNoise(n_fft // 2 + 1, hop / sr), hop / sr, rule=rule)
    y = process_signal(x, Wola(n_fft, hop), reducer)
    on = gate & (np.arange(len(x)) > sr)
    assert abs(10 * np.log10(np.mean(y[on] ** 2) / np.mean(tone[on] ** 2))) < 1


def test_si_sdr_ignores_gain_and_measures_noise():
    rng = np.random.default_rng(7)
    s, n = rng.standard_normal(16000), rng.standard_normal(16000)
    assert si_sdr(s, 0.3 * s + 1e-9 * n) > 100            # a scaled copy is perfect
    assert abs(si_sdr(s, s + 0.1 * n) - 20) < 0.5         # noise 20 dB below


def test_oracle_removes_noise_where_it_dominates():
    sr, n_fft, hop = 16000, 256, 128
    rng = np.random.default_rng(8)
    t = np.arange(sr) / sr
    clean = np.sin(2 * np.pi * 500 * t)
    noise = 0.3 * rng.standard_normal(sr)
    oracle = OracleWiener(clean, noise, n_fft, hop, gain_floor_db=-40)
    y = process_signal(clean + noise, Wola(n_fft, hop), oracle)
    assert si_sdr(clean, y) > si_sdr(clean, clean + noise) + 15
