"""Train the ESC-50 CNN with 5-fold cross-validation and save its out-of-fold predictions.

Usage (from the repository root):
    python scripts/train_cnn_esc50.py NAME [--global-frequency] [--no-augment] [--epochs 40]

The results are saved to data/results/NAME.npz and analyzed in notebooks/04_esc50_cnn.ipynb.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from audiolab.deep import augment_spectrogram, cnn_predict_on_folds
from audiolab.esc50 import compute_features, load_metadata
from audiolab.features import extract_logmel
from audiolab.model import AudioCNN, count_parameters

DATA = Path(__file__).resolve().parents[1] / "data"

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--global-frequency", action="store_true", help="average over frequency too")
parser.add_argument("--no-augment", action="store_true")
parser.add_argument("--epochs", type=int, default=40)
args = parser.parse_args()

meta = load_metadata(DATA / "ESC-50")
S = compute_features(DATA / "ESC-50", meta.filename, extract_logmel, sr=22050,
                     cache=DATA / "cache" / "esc50_logmel128_22k.npy")
y = meta.target.to_numpy()
folds = meta.fold.to_numpy()

config = dict(keep_frequency=not args.global_frequency, augment=not args.no_augment, epochs=args.epochs)
make_model = lambda: AudioCNN(n_classes=50, n_freq=S.shape[1], keep_frequency=config["keep_frequency"])
print(f"{args.name}: {config} | {count_parameters(make_model()):,} weights", flush=True)

start = time.time()
pred, histories = cnn_predict_on_folds(
    make_model, S, y, folds, epochs=args.epochs,
    augment=augment_spectrogram if config["augment"] else None,
    save_dir=DATA / "models" / args.name,
)
accs = [np.mean(pred[folds == k] == y[folds == k]) for k in np.unique(folds)]
print(f"{args.name}: {np.mean(accs):.1%} ± {np.std(accs):.1%} in {(time.time() - start) / 60:.0f} min", flush=True)

(DATA / "results").mkdir(exist_ok=True)
np.savez(DATA / "results" / f"{args.name}.npz", pred=pred, y=y, folds=folds,
         losses=np.array([histories[k] for k in sorted(histories)]),
         config=json.dumps(config), n_weights=count_parameters(make_model()))
