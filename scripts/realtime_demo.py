"""Classify the sounds around you in real time with the ESC-50 CNN.

Every HOP seconds the latest WINDOW seconds of microphone audio go through the same pipeline as in
training (22.05 kHz, 128-band log-mel, normalization) and the CNN; the top 3 classes are shown.

The CNN always answers one of its 50 classes, even for silence, so a level gate decides first whether
there is any sound event: the background level is measured while the buffer fills (stay quiet then),
and a window is classified only if some 50 ms frame of it is MARGIN dB louder than that background.
Below MIN_PROB the top class is shown as 'not sure'. The defaults (2 s window, one model, 0.4) were chosen
on recorded sessions in notebooks/06_realtime.ipynb; comfort noise is off by default because it did not help.

Usage (from the repository root):
    python scripts/realtime_demo.py [--window 2] [--hop 0.5] [--alpha 0.5] [--margin 10] [--comfort-db -70]
                                    [--min-prob 0.4] [--ensemble] [--device N] [--duration S] [--log FILE.csv]
    python scripts/realtime_demo.py --list-devices
With --log, the session audio is also saved next to the log (FILE.wav), so that it can be replayed
offline with other settings (scripts/evaluate_home.py). Ctrl+C stops it.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from audiolab.audio_io import save_audio
from audiolab.realtime import (RingBuffer, WindowClassifier, background_level_db, class_probabilities, load_models,
                               load_packaged_model)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "esc50_cnn_noise.pt"     # included in the repository
FOLD_MODELS = ROOT / "data" / "models" / "cnn_noise"   # the 5 fold models, after training them locally

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=float, default=2.0, help="seconds of audio in each prediction")
parser.add_argument("--hop", type=float, default=0.5, help="seconds between predictions")
parser.add_argument("--alpha", type=float, default=0.5, help="smoothing: 1 = none, smaller = smoother")
parser.add_argument("--margin", type=float, default=10.0, help="dB above the background to count as an event")
parser.add_argument("--comfort-db", type=float, default=None, help="add pink comfort noise at this dBFS level")
parser.add_argument("--min-prob", type=float, default=0.4, help="show 'not sure' below this top-1 probability")
parser.add_argument("--ensemble", action="store_true",
                    help="average the 5 fold models (needs scripts/train_cnn_esc50.py cnn_noise --noise first)")
parser.add_argument("--device", type=int, default=None, help="input device (see --list-devices)")
parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
parser.add_argument("--log", default=None, help="save every prediction to this CSV file (and the audio as .wav)")
parser.add_argument("--list-devices", action="store_true")
args = parser.parse_args()

if args.list_devices:
    print(sd.query_devices())
    sys.exit()

models, classes = load_packaged_model(MODEL)
classes = np.array(classes)
if args.ensemble:
    models = load_models(FOLD_MODELS, folds=range(1, 6))

sr = int(sd.query_devices(args.device, "input")["default_samplerate"])   # the microphone's own rate
n_window = int(args.window * sr)
buffer = RingBuffer(n_window)
lock = threading.Lock()         # the audio thread writes while the main thread reads
received = 0                    # samples received so far
overflows = 0                   # blocks the audio thread could not deliver in time
recording = []                  # every block, to save the session audio (only with --log)


def callback(indata, frames, time_info, status):
    """Called by the audio thread for every block (~10-20 ms): only store it, no heavy work here."""
    global received, overflows
    if status.input_overflow:
        overflows += 1
    with lock:
        buffer.write(indata[:, 0])
    if args.log:
        recording.append(indata[:, 0].copy())   # indata is reused by the next block: copy it
    received += frames


def show(header, p, event, first):
    """Draw the top 3 classes as bars (or why there are none), overwriting the previous drawing."""
    lines = [header]
    sure = p.max() >= args.min_prob
    for rank, i in enumerate(np.argsort(p)[::-1][:3]):
        if event and sure:
            lines.append(f"  {classes[i]:<18} {'█' * round(30 * p[i]):<30} {p[i]:4.0%}")
        elif rank == 0:
            lines.append("  (no sound event)" if not event else f"  (not sure: {classes[i]}? {p[i]:.0%})")
        else:
            lines.append("")
    if not first:
        print(f"\x1b[{len(lines)}F", end="")          # ANSI: cursor up to the start of the drawing
    print("\n".join(line + "\x1b[K" for line in lines), flush=True)   # \x1b[K clears the rest of the line


os.system("")                   # enables ANSI escape codes in the Windows console
sys.stdout.reconfigure(encoding="utf-8")
class_probabilities(models, np.zeros(int(args.window * 22050), np.float32))   # warm-up: the 1st call is slow
if args.log:
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
log = open(args.log, "w") if args.log else None
if log:
    log.write("time,level_db,peak_db,event,compute_ms,top1,p1,top2,p2,top3,p3\n")

print(f"cnn_noise ({len(models)} model{'s' if len(models) > 1 else ''}) | microphone at {sr} Hz | "
      f"window {args.window} s, hop {args.hop} s, alpha {args.alpha}, comfort noise {args.comfort_db} dBFS"
      f" | Ctrl+C to stop")
with sd.InputStream(device=args.device, channels=1, samplerate=sr, callback=callback) as stream:
    print(f"Measuring the background for {args.window} s: stay quiet...")
    while received < n_window:  # no prediction until the window is full of real audio (no digital silence)
        time.sleep(0.05)
    with lock:
        background_db = background_level_db(buffer.read_last(n_window), sr)
    print(f"Background {background_db:.1f} dBFS -> events above {background_db + args.margin:.1f} dBFS")
    classify = WindowClassifier(models, sr, background_db, margin_db=args.margin, alpha=args.alpha,
                                comfort_db=args.comfort_db)

    start = next_tick = time.perf_counter()
    first = True
    try:
        while args.duration is None or time.perf_counter() - start < args.duration:
            time.sleep(max(0.0, next_tick - time.perf_counter()))
            next_tick = max(next_tick + args.hop, time.perf_counter())   # if late, skip ticks, don't burst

            t0 = time.perf_counter()
            with lock:
                y = buffer.read_last(n_window)
            p, event, levels = classify(y)
            compute_ms = 1000 * (time.perf_counter() - t0)

            t = time.perf_counter() - start
            level_db = levels[-round(args.hop * 20):].max()           # loudest frame of the latest hop
            header = (f"{t:7.1f} s | level {level_db:6.1f} dBFS (background {background_db:.0f}) | "
                      f"audio in {1000 * stream.latency:.0f} ms + compute {compute_ms:.0f} ms | overflows {overflows}")
            show(header, p, event, first)
            first = False
            if log:
                top = np.argsort(p)[::-1][:3]
                log.write(f"{t:.2f},{level_db:.1f},{levels.max():.1f},{int(event)},{compute_ms:.1f},"
                          + ",".join(f"{classes[i]},{p[i]:.3f}" for i in top) + "\n")
    except KeyboardInterrupt:
        pass
if log:
    log.close()
    save_audio(Path(args.log).with_suffix(".wav"), np.concatenate(recording), sr)
    print(f"Saved {args.log} and its audio ({received / sr:.0f} s)")
