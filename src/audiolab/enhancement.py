"""Noise reduction in the STFT domain, frame by frame, as a hearing aid has to run it.

The framework is weighted overlap-add (WOLA): every hop of R new samples, the latest N samples are
windowed, transformed with an FFT, multiplied by a gain per frequency bin, transformed back, windowed
again and overlap-added into the output. With the identity gain the output is exactly the input,
delayed N - R samples (perfect reconstruction).
"""
import numpy as np


def sqrt_hann(n):
    """Square root of a periodic Hann window of n samples.

    Used for both analysis and synthesis, so their product is a Hann window, whose copies shifted by
    n/2 (or n/4) add up to a constant: the condition for perfect reconstruction. Splitting it in two
    square roots also tapers the synthesis frames, so a gain that changes from one frame to the next
    does not leave discontinuities at the frame edges.
    """
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))


class Wola:
    """Streaming STFT analysis -> gain -> synthesis with weighted overlap-add.

    n_fft: frame length N in samples; hop: R new samples per frame (n_fft // 2 or n_fft // 4).
    Call process(block, fn) with exactly `hop` new input samples; fn receives the spectrum of the
    latest frame (n_fft // 2 + 1 complex bins) and returns the modified spectrum. It returns `hop`
    output samples, which are the input delayed by n_fft - hop samples (self.delay).
    """

    def __init__(self, n_fft, hop):
        if n_fft % hop:
            raise ValueError("hop must divide n_fft")
        self.n_fft, self.hop = n_fft, hop
        self.window = sqrt_hann(n_fft)
        # Copies of window**2 (a Hann) shifted by the hop add up to sum(window**2) / hop: normalize to 1
        self.scale = hop / np.sum(self.window ** 2)
        self.frame = np.zeros(n_fft)          # the latest n_fft input samples, oldest first
        self.acc = np.zeros(n_fft)            # overlap-add accumulator; acc[:hop] is complete each hop
        self.delay = n_fft - hop

    def process(self, block, fn=None):
        """Feed `hop` new samples, get `hop` output samples. fn=None leaves the spectrum unchanged."""
        R = self.hop
        if len(block) != R:
            raise ValueError(f"expected {R} samples, got {len(block)}")
        self.frame[:-R] = self.frame[R:]      # shift the frame left by one hop (like buffer(x, N, N-R) in MATLAB)
        self.frame[-R:] = block

        X = np.fft.rfft(self.window * self.frame)
        if fn is not None:
            X = fn(X)
        self.acc += self.scale * self.window * np.fft.irfft(X, self.n_fft)

        out = self.acc[:R].copy()             # no later frame will add to these samples any more
        self.acc[:-R] = self.acc[R:]
        self.acc[-R:] = 0
        return out


def process_signal(x, wola, fn=None):
    """Run a whole signal through a Wola, block by block, exactly as in real time.

    The output is realigned with the input (the n_fft - hop samples of delay are removed), so it can
    be compared sample by sample with a clean reference. Same length as x.
    """
    R, D = wola.hop, wola.delay
    n_blocks = -(-(len(x) + D) // R)          # ceil: enough blocks to flush the delayed tail out
    padded = np.zeros(n_blocks * R)
    padded[:len(x)] = x
    out = np.concatenate([wola.process(padded[i * R:(i + 1) * R], fn) for i in range(n_blocks)])
    return out[D:D + len(x)]


def frame_sizes(frame_ms, sr, overlap=2):
    """(n_fft, hop) for a frame of frame_ms milliseconds; overlap 2 = 50%, 4 = 75%."""
    n_fft = int(round(frame_ms * sr / 1000))
    return n_fft, n_fft // overlap


REF_HOP_S = 0.016       # the published smoothing constants below are for 16 ms hops


def per_hop(alpha_ref, hop_s):
    """Smoothing constant for hop_s with the same time constant as alpha_ref at a 16 ms hop.

    A recursive average p = alpha p + (1 - alpha) x forgets like exp(-t / tau), with
    alpha = exp(-hop / tau). With shorter hops there are more updates per second, so alpha has to
    get closer to 1 for the average to cover the same time.
    """
    return alpha_ref ** (hop_s / REF_HOP_S)


class MCRA:
    """Noise power estimation by minima controlled recursive averaging (Cohen and Berdugo, 2002).

    Idea: the smoothed power of a bin rarely falls below the noise level, and speech comes and goes
    faster than the noise changes, so the minimum over the last ~1.5 s is a speech-free reference.
    Where the power is far above that minimum (ratio > delta) speech is probably present and the
    noise estimate is frozen; elsewhere it follows the noisy power with a slow recursive average.
    Only the decision uses the minimum, so its downward bias does not reach the noise estimate.
    The first init_s seconds are assumed to be noise only (their average is the initial estimate),
    like a device that is switched on before anyone speaks.
    """

    def __init__(self, n_bins, hop_s, window_s=1.5, init_s=0.1, alpha_s=0.8, alpha_p=0.2,
                 alpha_d=0.95, delta=5.0):
        self.alpha_s = per_hop(alpha_s, hop_s)        # smoothing of the power before taking minima
        self.alpha_p = per_hop(alpha_p, hop_s)        # smoothing of the speech presence probability
        self.alpha_d = per_hop(alpha_d, hop_s)        # noise averaging where there is no speech
        self.delta = delta
        self.L = max(1, round(window_s / 2 / hop_s))  # minima over 2 sub-windows of L frames
        self.n_init = max(1, round(init_s / hop_s))
        self.count = 0
        self.S = self.S_min = self.S_tmp = self.noise = None
        self.p = np.zeros(n_bins)

    def update(self, power):
        """Feed |Y|^2 of a new frame; returns the noise power estimate per bin."""
        # Smooth across 3 neighbouring bins, then over time
        Sf = np.convolve(power, [0.25, 0.5, 0.25], mode="same")
        self.count += 1
        if self.count <= self.n_init:                 # start-up: plain average of the first frames
            k = self.count
            self.S = Sf if k == 1 else self.S + (Sf - self.S) / k
            self.noise = power.copy() if k == 1 else self.noise + (power - self.noise) / k
            self.S_min, self.S_tmp = self.S.copy(), self.S.copy()
            return self.noise
        self.S = self.alpha_s * self.S + (1 - self.alpha_s) * Sf

        # Running minimum: S_min is the minimum since the start of the previous sub-window,
        # S_tmp of the current one; every L frames the old sub-window is forgotten
        if self.count % self.L == 0:
            self.S_min = np.minimum(self.S_tmp, self.S)
            self.S_tmp = self.S.copy()
        else:
            self.S_min = np.minimum(self.S_min, self.S)
            self.S_tmp = np.minimum(self.S_tmp, self.S)

        speech = self.S > self.delta * self.S_min
        self.p = self.alpha_p * self.p + (1 - self.alpha_p) * speech
        a = self.alpha_d + (1 - self.alpha_d) * self.p  # p = 1 -> a = 1: the noise estimate is frozen
        self.noise = a * self.noise + (1 - a) * power
        return self.noise


class SPPNoise:
    """Noise power estimation with the speech presence probability (Gerkmann and Hendriks, 2012).

    For every bin, the probability that speech is present follows from comparing the new power with
    the previous noise estimate (gamma), assuming that when speech is there it is xi_h1 above the noise:
        P(speech | Y) = 1 / (1 + (1 + xi_h1) exp(-gamma xi_h1 / (1 + xi_h1)))
    The noise power of the frame is the expected |N|^2 given that probability,
        (1 - P) |Y|^2 + P noise,
    so a bin that is surely speech keeps the old estimate and a bin that is surely noise takes the new
    power. No minimum is tracked, so there is no window to tune and no bias to compensate.
    If a bin looks like speech for too long (P > 0.99 on average) P is capped: otherwise a sudden rise
    of the noise would be taken for speech forever.
    """

    def __init__(self, n_bins, hop_s, init_s=0.1, xi_h1_db=15.0, alpha=0.8, alpha_p=0.9):
        self.xi_h1 = 10 ** (xi_h1_db / 10)
        self.alpha = per_hop(alpha, hop_s)
        self.alpha_p = per_hop(alpha_p, hop_s)
        self.n_init = max(1, round(init_s / hop_s))
        self.count = 0
        self.noise = None
        self.p_avg = np.zeros(n_bins)
        self.p = np.zeros(n_bins)

    def update(self, power):
        """Feed |Y|^2 of a new frame; returns the noise power estimate per bin."""
        self.count += 1
        if self.count <= self.n_init:                 # start-up: plain average of the first frames
            k = self.count
            self.noise = power.copy() if k == 1 else self.noise + (power - self.noise) / k
            return self.noise
        gamma = power / np.maximum(self.noise, 1e-12)
        x = self.xi_h1 / (1 + self.xi_h1)
        p = 1 / (1 + (1 + self.xi_h1) * np.exp(-np.minimum(gamma * x, 500)))  # cap: exp overflows
        self.p_avg = self.alpha_p * self.p_avg + (1 - self.alpha_p) * p
        p = np.where(self.p_avg > 0.99, np.minimum(p, 0.99), p)
        self.p = p
        expected = (1 - p) * power + p * self.noise
        self.noise = self.alpha * self.noise + (1 - self.alpha) * expected
        return self.noise


class NoiseReducer:
    """Gain per bin from the noisy spectrum and a noise estimate: pass it as fn to Wola.process.

    rule:
      "subtraction"  power spectral subtraction, |S|^2 = |Y|^2 - noise -> G = sqrt(1 - 1/gamma)
      "wiener"       Wiener gain xi / (1 + xi) with the SNR of the current frame only, xi = gamma - 1
      "wiener_dd"    Wiener gain with the decision-directed SNR estimate (Ephraim and Malah, 1984):
                     xi mixes the SNR of the previous frame's output with the current one
    gamma = |Y|^2 / noise is the a posteriori SNR, xi the a priori SNR (clean speech / noise).
    The gain never goes below gain_floor_db: residual noise sounds natural instead of musical,
    and a hearing aid user still hears the environment.
    noise: an estimator with update(power) -> noise power (MCRA), or a fixed array of noise powers.
    """

    def __init__(self, noise, hop_s, rule="wiener_dd", gain_floor_db=-15.0, alpha_dd=0.98):
        self.noise, self.rule = noise, rule
        self.g_min = 10 ** (gain_floor_db / 20)
        self.alpha_dd = per_hop(alpha_dd, hop_s)
        self.prev_clean = None                        # |G Y|^2 of the previous frame

    def __call__(self, X):
        power = np.abs(X) ** 2
        noise = self.noise.update(power) if hasattr(self.noise, "update") else self.noise
        gamma = power / np.maximum(noise, 1e-12)
        xi_ml = np.maximum(gamma - 1, 0)

        if self.rule == "subtraction":
            G = np.sqrt(xi_ml / np.maximum(gamma, 1e-12))
        elif self.rule == "wiener":
            G = xi_ml / (1 + xi_ml)
        elif self.rule == "wiener_dd":
            if self.prev_clean is None:
                xi = xi_ml
            else:
                xi = (self.alpha_dd * self.prev_clean / np.maximum(noise, 1e-12)
                      + (1 - self.alpha_dd) * xi_ml)
            G = xi / (1 + xi)
        else:
            raise ValueError(f"unknown rule {self.rule!r}")

        G = np.maximum(G, self.g_min)
        self.prev_clean = (G ** 2) * power
        return G * X


class OracleWiener:
    """The Wiener gain computed from the true speech and noise of every frame: an upper bound.

    No real system knows them, but it shows the best that a gain per bin can do with a given frame
    length, separately from the errors of estimating the noise. Use with a fresh Wola of the same
    n_fft and hop, on the noisy signal clean + noise.
    """

    def __init__(self, clean, noise, n_fft, hop, gain_floor_db=-15.0):
        S, N = (np.abs(stft_frames(x, n_fft, hop)) ** 2 for x in (clean, noise))
        self.gains = iter(np.maximum(S / np.maximum(S + N, 1e-12), 10 ** (gain_floor_db / 20)))

    def __call__(self, X):
        return next(self.gains) * X


def stft_frames(x, n_fft, hop):
    """The spectra of all the frames that process_signal would see, as an array (frames, bins)."""
    frames = []
    process_signal(x, Wola(n_fft, hop), lambda X: frames.append(X) or X)
    return np.array(frames)


def si_sdr(reference, estimate):
    """Scale-invariant signal-to-distortion ratio in dB (Le Roux et al., 2019).

    The estimate is split into a scaled copy of the reference (the projection onto it) and the rest,
    and SI-SDR is the power ratio of the two. A global gain does not change it; noise, removed speech
    and artifacts all count as distortion.
    """
    a = np.dot(estimate, reference) / np.dot(reference, reference)
    target = a * reference
    return 10 * np.log10(np.sum(target ** 2) / np.sum((estimate - target) ** 2))
