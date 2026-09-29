# audiolab

[![tests](https://github.com/victor-merino-hub/audiolab/actions/workflows/tests.yml/badge.svg)](https://github.com/victor-merino-hub/audiolab/actions/workflows/tests.yml)

Audio signal processing and machine learning for **hearables**: from DSP fundamentals to
sound classification and noise reduction under real-time constraints.

> Work in progress. I am a Telecommunications Engineering student building this project
> step by step to learn audio DSP and ML in depth. See the [roadmap](#roadmap).

## What's inside

| Module | What it does |
|---|---|
| `generator` | Periodic signals (sine, square, triangle, sawtooth) with aliasing checks |
| `analysis` | FFT spectrum and spectrogram |
| `processing` | Speed change vs. time-stretch (phase vocoder) |
| `audio_io` | Load, save and record audio |
| `dataset` | Labeled synthetic datasets with a controlled SNR |
| `features` | MFCC and harmonic-structure features |
| `model`, `deep` | CNN on spectrograms (PyTorch) |

The notebook [`01_fundamentals.ipynb`](notebooks/01_fundamentals.ipynb) walks through all of
them. One result from it: a RandomForest classifying waveforms reaches ~0.89 accuracy with
generic MFCC features, but **1.00 with 8 hand-crafted features** based on the Fourier series of
each waveform (square: odd harmonics at 1/k, triangle: 1/k², sawtooth: all at 1/k). Domain
knowledge beats generic features when the problem is well understood.

The tests check physical properties rather than just "it runs": the spectrum of a sine peaks
at ±f with amplitude A/2, the harmonic ratios match the Fourier series, the measured SNR
matches the requested one, and time-stretching keeps the pitch.

## Roadmap

- [x] **DSP fundamentals**: signal generation, spectrum, spectrogram, time-stretch
- [x] **Project setup**: installable package, tests, CI
- [ ] **Environmental sound classification on real data** (ESC-50): MFCC + RandomForest baseline,
      CNN on log-mel spectrograms, 5-fold cross-validation
- [ ] **Real-time demo**: classify live microphone input frame by frame
- [ ] **Noise reduction**: spectral subtraction and Wiener filtering, then a small mask-estimation
      network, evaluated with SNR, PESQ and STOI
- [ ] **Hearable constraints**: latency budget, model size, quantization / ONNX export

## Project structure

```
src/audiolab/   the package
notebooks/      experiments and explanations
tests/          pytest test suite
data/           audio files (not committed)
```

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/victor-merino-hub/audiolab.git
cd audiolab
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

```python
from audiolab.generator import generate_signal
from audiolab.analysis import compute_spectrum

t, y = generate_signal("square", frequency=440, amplitude=0.5, fs=44100, duration=1.0)
f, P = compute_spectrum(y, fs=44100)
```

## License

[MIT](LICENSE)
