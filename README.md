# audiolab

[![tests](https://github.com/victor-merino-hub/audiolab/actions/workflows/tests.yml/badge.svg)](https://github.com/victor-merino-hub/audiolab/actions/workflows/tests.yml)

Audio signal processing and machine learning for **hearables**: from DSP fundamentals to
sound classification and noise reduction under real-time constraints.

> Work in progress. I am a Telecommunications Engineering student building this project
> step by step to learn audio DSP and ML in depth. See the [roadmap](#roadmap).

## Results so far

### Environmental sound classification (ESC-50)

[ESC-50](https://github.com/karolpiczak/ESC-50): 2,000 real clips of 5 s in 50 classes, evaluated
with the 5 official cross-validation folds. Chance level is 2%.

| Model | Weights | Accuracy | Reference |
|---|---|---|---|
| MFCC statistics + Random Forest | - | 47.6% ± 2.1% | 44.3% in the ESC-50 paper |
| **CNN on log-mel spectrograms** | 149k | **76.5% ± 2.7%** | 64.5% in Piczak (2015) |
| Human listeners | | | 81.3% |

![Accuracy of each model on the 5 official folds](docs/figures/esc50_models.png)

The CNN (4 conv blocks, trained on a laptop CPU in 47 minutes) wins most where the baseline
was blind: averaging MFCCs over time destroys temporal patterns, so **clock tick goes from 5% to 80%**
and car horn from 10% to 82%. Two design experiments, with everything else fixed:

- **Keep the frequency position** (average only over time at the end): +13.6 points. Time shifts
  should not change the class, but frequency position does: a steady tone is a car horn or an
  alarm depending on where it sits. Without it the network cannot even fit its training set.
- **Data augmentation** (time shift + SpecAugment): no measurable gain here (76.5% vs 76.2%). The
  time shift is redundant with a network that already averages over time. For scale: retraining
  the same model on a GPU instead of the CPU changes 9% of the predictions and the mean by 0.5
  points, so differences below ~1 point are noise.

**Sanity checks** before trusting a result this far above the reference: training on shuffled
labels gives 2.2% (chance: 2%), so nothing leaks from the test folds; and the class-dependent
band-limiting the dataset authors warn about is a weak cue (3.9% on its own). The gap with the 2015
CNN is mostly a smaller model (149k vs ~26M weights), whole clips instead of 1 s segments, and batch
normalization.

Notebooks: [exploration](notebooks/02_esc50_exploration.ipynb) · [baseline and leakage](notebooks/03_esc50_baseline.ipynb) · [CNN](notebooks/04_esc50_cnn.ipynb) · [robustness](notebooks/05_esc50_robustness.ipynb)

### The digital silence shortcut

ESC-50 pads short clips to 5 s with exact zeros, more for some classes (glass breaking, sneezing)
than others, and the network learns to use it. A real microphone never outputs exact zeros: adding
pink background noise only 60 dB below the sound leaves the clips without padding untouched
(77.3% → 77.3%) but costs the padded ones 9 points, and 42 points at 20 dB. The benchmark number
is optimistic for short, impulsive sounds in the real world.

![Accuracy with background noise, padded vs. not padded clips](docs/figures/silence_shortcut.png)

### Data leakage: how a random split lies

40% of the ESC-50 clips were cut from the same original recording as another clip, and share its
microphone, room and background noise. The official folds keep them together. Splitting at random
instead reports **58.2% instead of 47.6%**: 10.6 points that are not real. The gain appears only
in the clips whose "sibling" ended up in training (49% → 74%), while the rest barely change
(47% → 50%): the model is recognizing recordings, not sounds. Removing the microphone's
fingerprint with cepstral mean normalization halves the leakage (+4.7 points) but costs 16 points
of accuracy, since it also removes the timbre of continuous sounds.

![Accuracy with official vs. random folds](docs/figures/leakage.png)

For a hearable this matters: it has to work in every user's home, not on the recordings it was
trained on.

## What's inside

| Module | What it does |
|---|---|
| `generator` | Periodic signals (sine, square, triangle, sawtooth) with aliasing checks |
| `analysis` | FFT spectrum and spectrogram |
| `processing` | Speed change vs. time-stretch (phase vocoder), pink background noise at a given SNR |
| `audio_io` | Load, save and record audio |
| `dataset` | Labeled synthetic datasets with a controlled SNR |
| `features` | MFCC, log-mel spectrograms and harmonic-structure features |
| `model`, `deep` | CNN on spectrograms (PyTorch) |
| `esc50` | ESC-50 metadata and clip loading |
| `evaluation` | Cross-validation on predefined folds, random folds for comparison |

Training scripts live in [`scripts/`](scripts/): `train_cnn_esc50.py` trains a CNN configuration
with 5-fold cross-validation and saves its predictions for the analysis notebook.

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
- [x] **ESC-50 baseline**: MFCC + Random Forest over the 5 official folds
- [x] **Data leakage experiment**: random split vs. official folds
- [x] **CNN on log-mel spectrograms** (ESC-50), with error analysis
- [ ] **Pretrained audio model** (AudioSet embeddings): the accuracy ceiling, and how much a
      hearable-sized model gives up (accuracy vs. model size)
- [x] **Digital silence shortcut**: measure the accuracy drop when the padding is replaced by
      background noise
- [ ] **Noise augmentation**: train with background noise so the shortcut disappears
- [ ] **Real-time demo**: classify live microphone input frame by frame
- [ ] **Noise reduction**: spectral subtraction and Wiener filtering, then a small mask-estimation
      network, evaluated with SNR, PESQ and STOI
- [ ] **Hearable constraints**: latency budget, model size, quantization / ONNX export

## Project structure

```
src/audiolab/   the package
notebooks/      experiments and explanations
scripts/        training scripts
docs/figures/   figures used in this README
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

The ESC-50 notebooks expect the dataset (~600 MB) unzipped in `data/ESC-50/`: download it from
[github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50). It is licensed
CC BY-NC 3.0 by Karol J. Piczak and is not redistributed here.

Training a CNN takes ~47 minutes on a laptop CPU. To use a free GPU instead, open
[`gpu_training.ipynb`](notebooks/gpu_training.ipynb) in Google Colab or Kaggle: it clones this
repository, downloads ESC-50, trains, and packages the models for download.

```python
from audiolab.generator import generate_signal
from audiolab.analysis import compute_spectrum

t, y = generate_signal("square", frequency=440, amplitude=0.5, fs=44100, duration=1.0)
f, P = compute_spectrum(y, fs=44100)
```

## License

[MIT](LICENSE)
