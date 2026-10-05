"""A minimal hearing aid: microphone -> noise reduction -> headphones, live, with its latency measured.

USE WIRED HEADPHONES. With the laptop speakers the output reaches the microphone again and howls
(acoustic feedback, the whistle of a badly fitted hearing aid). Bluetooth headphones work but add
100-200 ms, which is exactly what a hearing aid cannot afford.

    python scripts/hearing_aid_demo.py                  # live; press Enter to switch the noise reduction on/off
    python scripts/hearing_aid_demo.py --frame-ms 32    # same with the textbook frame length
    python scripts/hearing_aid_demo.py --measure        # measure the latency of the audio chain (see below)
    python scripts/hearing_aid_demo.py --file in.wav    # process a file offline through the same chain

--measure plays clicks through the output and records them with the microphone (hold the headphones
against the microphone, or use the speakers at low volume for this test only). The delay between a
click being sent and coming back is the output + input latency of the sound card and drivers; adding
the frame length gives the total delay from the microphone to the ear.
"""
import argparse
import queue
import threading

import numpy as np

from audiolab.enhancement import MCRA, NoiseReducer, SPPNoise, Wola, frame_sizes

SR = 16000


class LiveChain:
    """What runs on every block of `hop` samples: noise reduction (switchable) and an output gain."""

    def __init__(self, frame_ms, overlap, rule, noise, floor_db, gain_db):
        self.n_fft, self.hop = frame_sizes(frame_ms, SR, overlap)
        self.wola = Wola(self.n_fft, self.hop)
        Estimator = SPPNoise if noise == "spp" else MCRA
        self.reducer = NoiseReducer(Estimator(self.n_fft // 2 + 1, self.hop / SR), self.hop / SR, rule,
                                    gain_floor_db=floor_db)
        self.gain = 10 ** (gain_db / 20)
        self.enabled = True

    def __call__(self, block):
        # The noise estimate keeps learning even when the reduction is off, so switching it on is instant;
        # off, the WOLA still runs with G = 1, so the delay is the same and only the gain changes
        out = self.wola.process(block, self._apply)
        return np.clip(self.gain * out, -1, 1)

    def _apply(self, X):
        Y = self.reducer(X)
        return Y if self.enabled else X


def run_live(chain, device):
    import sounddevice as sd

    status_q = queue.Queue()

    def callback(indata, outdata, frames, time, status):
        if status:
            status_q.put(str(status))           # under/overflows: never print inside the audio thread
        outdata[:, 0] = chain(indata[:, 0].astype(np.float64))

    with sd.Stream(samplerate=SR, blocksize=chain.hop, channels=1, dtype="float32", latency="low",
                   device=device, callback=callback) as stream:
        in_lat, out_lat = stream.latency
        algo = chain.n_fft / SR
        print(f"frame {chain.n_fft} samples ({1000 * algo:.0f} ms), hop {chain.hop}")
        print(f"reported latency: input {1000 * in_lat:.1f} ms + output {1000 * out_lat:.1f} ms "
              f"+ algorithm {1000 * algo:.0f} ms = {1000 * (in_lat + out_lat + algo):.1f} ms")
        print("Enter: noise reduction on/off · q + Enter: quit")
        stop = threading.Event()

        def keys():
            while not stop.is_set():
                if input().strip().lower() == "q":
                    stop.set()
                else:
                    chain.enabled = not chain.enabled
                    print("noise reduction", "ON" if chain.enabled else "OFF")

        threading.Thread(target=keys, daemon=True).start()
        while not stop.wait(0.5):
            while not status_q.empty():
                print("audio:", status_q.get())


def measure_latency(device, n_clicks=8, period=0.5):
    """Round-trip latency of the sound card: play clicks, record them, find the delay by cross-correlation."""
    import sounddevice as sd

    n = int(period * SR)
    click = np.zeros(n)
    click[:32] = np.hanning(32)                 # a short smooth pulse, broadband but without a hard edge
    out = np.tile(click, n_clicks) * 0.5
    rec = sd.playrec(out, samplerate=SR, channels=1, device=device, latency="low", blocking=True)[:, 0]

    delays = []
    for i in range(1, n_clicks):                # skip the first: the stream is still starting
        seg = rec[i * n:(i + 1) * n]
        corr = np.correlate(seg, click[:64], mode="valid")
        delays.append(np.argmax(np.abs(corr)) / SR)
        peak = np.max(np.abs(seg))
    delays = np.array(delays)
    print(f"round trip (output + input): median {1000 * np.median(delays):.1f} ms, "
          f"spread {1000 * (delays.max() - delays.min()):.1f} ms, last click peak {peak:.3f}")
    if peak < 0.01:
        print("the clicks were barely recorded: move the headphones closer to the microphone")
    return np.median(delays)


def run_file(chain, path, out_path):
    import soundfile as sf
    from scipy.signal import resample_poly

    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = resample_poly(y, SR, sr)
    n = len(y) // chain.hop * chain.hop
    out = np.concatenate([chain(y[i:i + chain.hop]) for i in range(0, n, chain.hop)])
    sf.write(out_path, out, SR)
    print(f"saved {out_path} ({len(out) / SR:.1f} s, delayed {1000 * chain.wola.delay / SR:.1f} ms)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frame-ms", type=float, default=8, help="frame length = algorithmic latency (default 8)")
    parser.add_argument("--overlap", type=int, default=2, help="2 = 50%%, 4 = 75%%")
    parser.add_argument("--rule", default="wiener_dd", choices=["subtraction", "wiener", "wiener_dd"])
    parser.add_argument("--noise", default="spp", choices=["spp", "mcra"])
    parser.add_argument("--floor-db", type=float, default=-15, help="maximum attenuation (gain floor)")
    parser.add_argument("--gain-db", type=float, default=0, help="output gain")
    parser.add_argument("--device", default=None, help="sounddevice device (index or name); default: system default")
    parser.add_argument("--measure", action="store_true", help="measure the round-trip latency and exit")
    parser.add_argument("--file", help="process this file offline instead of the microphone")
    parser.add_argument("--out", default="hearing_aid_out.wav", help="output file for --file")
    args = parser.parse_args()

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    if args.measure:
        rt = measure_latency(device)
        n_fft, _ = frame_sizes(args.frame_ms, SR, args.overlap)
        print(f"microphone to ear with {args.frame_ms:g} ms frames: {1000 * rt + 1000 * n_fft / SR:.1f} ms")
        return
    chain = LiveChain(args.frame_ms, args.overlap, args.rule, args.noise, args.floor_db, args.gain_db)
    if args.file:
        run_file(chain, args.file, args.out)
    else:
        run_live(chain, device)


if __name__ == "__main__":
    main()
