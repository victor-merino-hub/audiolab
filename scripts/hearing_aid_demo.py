"""A minimal hearing aid: microphone -> noise reduction -> headphones, live, with its latency measured.

USE WIRED HEADPHONES. With the laptop speakers the output reaches the microphone again and howls
(acoustic feedback, the whistle of a badly fitted hearing aid). Bluetooth headphones work but add
100-200 ms, which is exactly what a hearing aid cannot afford.

    python scripts/hearing_aid_demo.py                       # live; Enter switches the noise reduction on/off
    python scripts/hearing_aid_demo.py --frame-ms 32         # same with the textbook frame length
    python scripts/hearing_aid_demo.py --hostapi wasapi      # choose the Windows audio API (see below)
    python scripts/hearing_aid_demo.py --measure             # measure the latency of the audio chain
    python scripts/hearing_aid_demo.py --file in.wav         # process a file offline through the same chain

--measure plays clicks through the output and records them with the microphone (hold the headphones
against the microphone, or use the speakers at low volume for this test only). The delay between a
click being sent and coming back is the output + input latency of the sound card, the drivers and the
operating system; adding the frame length gives the total delay from the microphone to the ear.

--hostapi: Windows has several audio APIs and they buffer very differently. MME (the default) is the
oldest and buffers the most; WASAPI and WDM-KS go closer to the driver. wasapi-exclusive and wdm-ks
use the device's own sample rate (44.1 or 48 kHz) and the algorithm runs at that rate.
"""
import argparse
import queue
import threading

import numpy as np

from audiolab.enhancement import MCRA, NoiseReducer, SPPNoise, Wola, frame_sizes

SR = 16000              # the rate of the evaluation; used whenever the audio API can convert to it
HOSTAPIS = {"mme": "MME", "directsound": "Windows DirectSound", "wasapi": "Windows WASAPI",
            "wasapi-exclusive": "Windows WASAPI", "wdm-ks": "Windows WDM-KS"}


class LiveChain:
    """What runs on every block of `hop` samples: noise reduction (switchable) and an output gain."""

    def __init__(self, sr, frame_ms, overlap, rule, noise, floor_db, gain_db):
        self.sr = sr
        self.n_fft, self.hop = frame_sizes(frame_ms, sr, overlap)
        self.wola = Wola(self.n_fft, self.hop)
        Estimator = SPPNoise if noise == "spp" else MCRA
        self.reducer = NoiseReducer(Estimator(self.n_fft // 2 + 1, self.hop / sr), self.hop / sr, rule,
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


def audio_setup(hostapi, device):
    """(device, extra_settings, sample rate) for a host API, using its default input and output devices."""
    import sounddevice as sd

    if hostapi is None:
        return device, None, SR
    names = [h["name"] for h in sd.query_hostapis()]
    if HOSTAPIS[hostapi] not in names:
        raise SystemExit(f"{HOSTAPIS[hostapi]} is not available here: {', '.join(names)}")
    api = sd.query_hostapis(names.index(HOSTAPIS[hostapi]))
    device = (api["default_input_device"], api["default_output_device"])
    if hostapi == "wasapi":                       # shared mode: Windows converts to and from 16 kHz
        settings = sd.WasapiSettings(auto_convert=True)
        return device, (settings, settings), SR
    sr = int(sd.query_devices(device[1])["default_samplerate"])
    if hostapi == "wasapi-exclusive":
        settings = sd.WasapiSettings(exclusive=True)
        return device, (settings, settings), sr
    if hostapi == "wdm-ks":
        return device, None, sr
    return device, None, SR


def describe(device):
    import sounddevice as sd

    ins, outs = device if isinstance(device, tuple) else (sd.default.device[0], sd.default.device[1])
    api = lambda d: sd.query_hostapis(sd.query_devices(d)["hostapi"])["name"]
    return (f"input: {sd.query_devices(ins)['name']} ({api(ins)}) · "
            f"output: {sd.query_devices(outs)['name']} ({api(outs)})")


def run_live(chain, device, extra, record=None):
    """Run the chain on the microphone. record: a path prefix to save, when the session ends, the input
    and output audio (<prefix>_in.wav, <prefix>_out.wav) and a log of every block (<prefix>_log.csv:
    compute time, under/overflows, noise reduction on or off), to find out what went wrong offline."""
    import time as clock

    import sounddevice as sd

    status_q = queue.Queue()
    log = {"in": [], "out": [], "compute_ms": [], "status": [], "enabled": []}

    def callback(indata, outdata, frames, time, status):
        start = clock.perf_counter()
        if status:
            status_q.put(str(status))           # under/overflows: never print inside the audio thread
        outdata[:, 0] = chain(indata[:, 0].astype(np.float64))
        if record:                              # list appends are cheap enough for the audio thread
            log["in"].append(indata[:, 0].copy())
            log["out"].append(outdata[:, 0].copy())
            log["compute_ms"].append(1000 * (clock.perf_counter() - start))
            log["status"].append(str(status) if status else "")
            log["enabled"].append(chain.enabled)

    with sd.Stream(samplerate=chain.sr, blocksize=chain.hop, channels=1, dtype="float32", latency="low",
                   device=device, extra_settings=extra, callback=callback) as stream:
        in_lat, out_lat = stream.latency
        algo = chain.n_fft / chain.sr
        print(describe(device))
        print(f"{chain.sr} Hz, frame {chain.n_fft} samples ({1000 * algo:.1f} ms), hop {chain.hop}")
        print(f"reported latency: input {1000 * in_lat:.1f} ms + output {1000 * out_lat:.1f} ms "
              f"+ algorithm {1000 * algo:.1f} ms = {1000 * (in_lat + out_lat + algo):.1f} ms")
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
    if record and log["in"]:
        save_session(log, chain, record)


def save_session(log, chain, prefix):
    import pandas as pd
    import soundfile as sf

    sf.write(f"{prefix}_in.wav", np.concatenate(log["in"]), chain.sr)
    sf.write(f"{prefix}_out.wav", np.concatenate(log["out"]), chain.sr)
    blocks = pd.DataFrame({"compute_ms": log["compute_ms"], "status": log["status"], "enabled": log["enabled"]})
    blocks.to_csv(f"{prefix}_log.csv", index_label="block")
    hop_ms = 1000 * chain.hop / chain.sr
    print(f"saved {prefix}_in.wav, _out.wav, _log.csv: {len(blocks)} blocks of {hop_ms:.1f} ms, "
          f"compute median {blocks.compute_ms.median():.2f} ms, max {blocks.compute_ms.max():.2f} ms, "
          f"{(blocks.status != '').sum()} blocks with under/overflows")


def chirp(sr, duration=0.02, f0=500, f1=6000):
    """Linear frequency sweep with tapered ends: its autocorrelation is one narrow peak (pulse compression)."""
    t = np.arange(int(duration * sr)) / sr
    f1 = min(f1, 0.45 * sr)
    return np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)) * np.hanning(len(t))


def measure_latency(device, extra, sr, n_probes=12, period=0.5, save=None):
    """Round-trip latency of the audio chain, like a radar: send a chirp every `period` seconds, record,
    and find the delay with a matched filter (correlation with the chirp).

    The probes are sent in the same place of every period, so averaging the periods before correlating
    adds the echo coherently and averages the noise out: ~10 dB more SNR with 11 periods. The delay must
    be shorter than the period. Quality = height of the correlation peak over its median: a peak that
    does not stand out (below ~8) is noise, not the probe.
    """
    import sounddevice as sd

    n = int(period * sr)
    probe = chirp(sr)
    frame = np.zeros(n)
    frame[:len(probe)] = probe
    out = np.tile(frame, n_probes) * 0.5
    print(describe(device), f"· {sr} Hz")
    rec = sd.playrec(out, samplerate=sr, channels=1, device=device, latency="low", extra_settings=extra,
                     blocking=True)[:, 0]
    if save:
        import soundfile as sf
        sf.write(save, rec, sr)

    periods = rec[n:n * n_probes].reshape(n_probes - 1, n)   # skip the first: the stream is still starting

    def detect(x):
        corr = np.abs(np.correlate(np.concatenate([x, x[:len(probe)]]), probe, mode="valid"))[:n]
        k = np.argmax(corr)
        return k / sr, corr[k] / np.median(corr)

    each = [detect(p) for p in periods]
    delay, quality = detect(periods.mean(axis=0))
    print("delay of each probe (ms):", " ".join(f"{1000 * d:.0f}" for d, _ in each))
    print(f"round trip (output + input), averaged: {1000 * delay:.1f} ms, detection quality {quality:.0f}, "
          f"peak level {np.abs(rec).max():.3f}")
    if quality < 8:
        print("UNRELIABLE: the probe does not stand out from the noise. Put the earphone right against the "
              "microphone and raise the volume.")
    return delay


def run_file(chain, path, out_path):
    import soundfile as sf
    from scipy.signal import resample_poly

    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != chain.sr:
        y = resample_poly(y, chain.sr, sr)
    n = len(y) // chain.hop * chain.hop
    out = np.concatenate([chain(y[i:i + chain.hop]) for i in range(0, n, chain.hop)])
    sf.write(out_path, out, chain.sr)
    print(f"saved {out_path} ({len(out) / chain.sr:.1f} s, delayed {1000 * chain.wola.delay / chain.sr:.1f} ms)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frame-ms", type=float, default=8, help="frame length = algorithmic latency (default 8)")
    parser.add_argument("--overlap", type=int, default=2, help="2 = 50%%, 4 = 75%%")
    parser.add_argument("--rule", default="wiener_dd", choices=["subtraction", "wiener", "wiener_dd", "wiener_tsnr"])
    parser.add_argument("--noise", default="spp", choices=["spp", "mcra"])
    parser.add_argument("--floor-db", type=float, default=-10, help="maximum attenuation (gain floor, default -10)")
    parser.add_argument("--gain-db", type=float, default=0, help="output gain")
    parser.add_argument("--hostapi", choices=list(HOSTAPIS), help="Windows audio API (default: the system default, MME)")
    parser.add_argument("--device", default=None, help="sounddevice device (index or name), without --hostapi")
    parser.add_argument("--measure", action="store_true", help="measure the round-trip latency and exit")
    parser.add_argument("--file", help="process this file offline instead of the microphone")
    parser.add_argument("--out", default="hearing_aid_out.wav", help="output file for --file")
    parser.add_argument("--record", metavar="PREFIX", help="save the live input, output and a block log (e.g. data/live)")
    args = parser.parse_args()

    if args.file:
        chain = LiveChain(SR, args.frame_ms, args.overlap, args.rule, args.noise, args.floor_db, args.gain_db)
        run_file(chain, args.file, args.out)
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    device, extra, sr = audio_setup(args.hostapi, device)
    if args.measure:
        rt = measure_latency(device, extra, sr, save=f"data/latency_{args.hostapi or 'default'}.wav")
        n_fft, _ = frame_sizes(args.frame_ms, sr, args.overlap)
        print(f"microphone to ear with {args.frame_ms:g} ms frames: {1000 * rt + 1000 * n_fft / sr:.1f} ms")
        return
    chain = LiveChain(sr, args.frame_ms, args.overlap, args.rule, args.noise, args.floor_db, args.gain_db)
    run_live(chain, device, extra, args.record)


if __name__ == "__main__":
    main()
