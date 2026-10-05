"""Evaluate the noise reduction on the VoiceBank+DEMAND test set (824 noisy sentences, 2 speakers).

Every method runs frame by frame exactly as in real time, for several frame lengths. Results, one row
per sentence and configuration, go to a CSV for the analysis notebook. PESQ is computed only if the
`pesq` package is installed (it needs a C compiler to install, so on Windows it runs in Colab).

Usage:
    python scripts/evaluate_denoising.py                      # all 824 sentences
    python scripts/evaluate_denoising.py --every 8            # 1 of every 8, for a quick check
    python scripts/evaluate_denoising.py --metrics pesq       # only PESQ (in Colab, where it installs)
"""
import argparse
import os

# One BLAS thread per worker process: the parallelism comes from the processes, and a process per
# core each starting a BLAS thread per core runs out of memory
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pystoi import stoi
from scipy.signal import resample_poly

from audiolab.enhancement import (MCRA, NoiseReducer, OracleWiener, SPPNoise, Wola, frame_sizes,
                                  process_signal, si_sdr)

try:
    from pesq import pesq
except ImportError:
    pesq = None

SR = 16000                      # VoiceBank+DEMAND is distributed at 48 kHz; the literature uses 16 kHz
FRAMES_MS = (4, 8, 16, 32)


def configurations():
    """(frame_ms, overlap, rule, noise estimator, gain floor in dB) of every method to evaluate."""
    configs = [(32, 2, "noisy", "-", 0.0)]                          # no processing: the reference
    for ms in FRAMES_MS:
        for rule in ("subtraction", "wiener", "wiener_dd", "wiener_tsnr"):
            configs.append((ms, 2, rule, "spp", -15.0))
        configs.append((ms, 2, "wiener_dd", "mcra", -15.0))
        configs.append((ms, 2, "oracle", "-", -15.0))
        configs.append((ms, 4, "wiener_dd", "spp", -15.0))          # 75% overlap, same latency
    for ms in (8, 32):
        for floor in (-6.0, -10.0, -25.0):                         # how much to suppress
            configs.append((ms, 2, "wiener_dd", "spp", floor))
    return configs


def enhance(clean, noisy, frame_ms, overlap, rule, noise, floor):
    if rule == "noisy":
        return noisy
    n_fft, hop = frame_sizes(frame_ms, SR, overlap)
    if rule == "oracle":
        fn = OracleWiener(clean, noisy - clean, n_fft, hop, floor)
    else:
        Estimator = SPPNoise if noise == "spp" else MCRA
        fn = NoiseReducer(Estimator(n_fft // 2 + 1, hop / SR), hop / SR, rule, gain_floor_db=floor)
    return process_signal(noisy, Wola(n_fft, hop), fn)


def evaluate_file(clean_path, metrics):
    noisy_path = Path(str(clean_path).replace("clean_testset_wav", "noisy_testset_wav"))
    clean, noisy = (resample_poly(sf.read(p)[0], 1, 3) for p in (clean_path, noisy_path))
    input_snr = 10 * np.log10(np.sum(clean ** 2) / np.sum((noisy - clean) ** 2))
    rows = []
    for frame_ms, overlap, rule, noise, floor in configurations():
        out = enhance(clean, noisy, frame_ms, overlap, rule, noise, floor)
        row = dict(file=Path(clean_path).stem, input_snr=input_snr, frame_ms=frame_ms,
                   overlap=overlap, rule=rule, noise=noise, floor_db=floor)
        if "si_sdr" in metrics:
            row["si_sdr"] = si_sdr(clean, out)
        if "stoi" in metrics:
            row["stoi"] = stoi(clean, out, SR)
        if "pesq" in metrics:
            row["pesq"] = pesq(SR, clean, out, "wb")            # wide-band PESQ (ITU-T P.862.2)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/voicebank", help="folder with clean_testset_wav/ and noisy_testset_wav/")
    parser.add_argument("--out", default="data/results/denoising.csv")
    parser.add_argument("--every", type=int, default=1, help="use 1 of every N sentences")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    parser.add_argument("--metrics", default="si_sdr,stoi,pesq",
                        help="comma-separated; pesq is skipped if the package is not installed")
    args = parser.parse_args()
    metrics = [m for m in args.metrics.split(",") if m != "pesq" or pesq is not None]

    files = sorted((Path(args.data) / "clean_testset_wav").glob("*.wav"))[::args.every]
    print(f"{len(files)} sentences x {len(configurations())} configurations, metrics: {', '.join(metrics)}"
          + ("" if pesq else " (PESQ not installed)"), flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, file_rows in enumerate(pool.map(partial(evaluate_file, metrics=metrics), files, chunksize=4), 1):
            rows += file_rows
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.out, index=False)
    keys = ["frame_ms", "overlap", "rule", "noise", "floor_db"]
    print(results.groupby(keys)[metrics].mean().round(3).to_string())
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
