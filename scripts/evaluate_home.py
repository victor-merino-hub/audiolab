"""Accuracy of the real-time demo on sounds recorded at home, replaying the sessions offline.

Each session is recorded live while producing a single sound repeatedly (stay quiet while the buffer fills):
    python scripts/realtime_demo.py --duration 40 --log data/home/door_wood_knock.csv
which saves the predictions (.csv) and the audio (.wav). This script replays every .wav of a folder
(default data/home) through the same code as the demo (realtime.replay), so different settings can be
compared on the same sounds:
    python scripts/evaluate_home.py [FOLDER] [--comfort-db -60] [--min-prob 0.4] [--window 5] [--ensemble] ...

The file name is the ESC-50 class (door_wood_knock_2.wav also works). Sessions of sounds that are NOT in
ESC-50 (speech, music...) are named other_<anything>.wav. Only windows with a sound event are scored.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audiolab.audio_io import load_audio
from audiolab.esc50 import load_metadata
from audiolab.realtime import load_models, replay

DATA = Path(__file__).resolve().parents[1] / "data"

parser = argparse.ArgumentParser()
parser.add_argument("sessions", nargs="?", default=DATA / "home", help="folder with the session .wav files")
parser.add_argument("--window", type=float, default=2.0)
parser.add_argument("--hop", type=float, default=0.5)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--margin", type=float, default=10.0)
parser.add_argument("--comfort-db", type=float, default=None)
parser.add_argument("--min-prob", type=float, default=0.4, help="a prediction is shown only above this top-1 probability")
parser.add_argument("--ensemble", action="store_true")
parser.add_argument("--model", default="cnn_noise")
args = parser.parse_args()

classes = load_metadata(DATA / "ESC-50").drop_duplicates("target").sort_values("target").category.to_numpy()
models = load_models(DATA / "models" / args.model, folds=range(1, 6) if args.ensemble else (1,))

rows = []
for path in sorted(Path(args.sessions).glob("*.wav")):
    name = path.stem
    true = next((c for c in sorted(classes, key=len, reverse=True) if name.startswith(c)), None)
    if true is None and not name.startswith("other"):
        print(f"Skipping {path.name}: not an ESC-50 class nor other_*")
        continue
    y, sr = load_audio(path)
    P, events = replay(y, sr, models, window=args.window, hop=args.hop, margin_db=args.margin,
                       alpha=args.alpha, comfort_db=args.comfort_db)
    P = P[events]                                          # only windows with a sound event
    ranked = np.argsort(P, axis=1)[:, ::-1]
    shown = P.max(axis=1) >= args.min_prob
    row = {"session": name, "windows": len(events), "events": len(P), "shown": shown.mean(),
           "mean_p1": P.max(axis=1).mean(), "most_predicted": pd.Series(classes[ranked[:, 0]]).mode().iat[0]}
    if true is not None:
        k = list(classes).index(true)
        row["top1_acc"] = (ranked[:, 0] == k).mean()
        row["top3_acc"] = (ranked[:, :3] == k).any(axis=1).mean()
        row["acc_shown"] = (ranked[shown, 0] == k).mean() if shown.any() else np.nan
    rows.append(row)

if not rows:
    raise SystemExit(f"No session recordings (.wav) in {args.sessions}")
table = pd.DataFrame(rows).set_index("session")
pd.set_option("display.width", 160)
print(table.round(2).to_string())
known = table.dropna(subset=["top1_acc"])
other = table.drop(known.index)
print(f"\n{vars(args)}")
print(f"ESC-50 classes: top-1 {known.top1_acc.mean():.0%}, top-3 {known.top3_acc.mean():.0%} "
      f"(ESC-50 cross-validation: 75%)", end="")
if args.min_prob > 0:
    print(f" | shown {known.shown.mean():.0%}, accuracy of shown {known.acc_shown.mean():.0%}"
          + (f" | other sounds shown {other.shown.mean():.0%}" if len(other) else ""), end="")
print()
