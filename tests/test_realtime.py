from pathlib import Path

import numpy as np

from audiolab.realtime import (RingBuffer, add_comfort_noise, class_probabilities, frame_levels_db,
                               load_packaged_model)

MODEL = Path(__file__).resolve().parents[1] / "models" / "esc50_cnn_noise.pt"


def test_read_after_one_write():
    buf = RingBuffer(10)
    buf.write(np.arange(4))
    np.testing.assert_array_equal(buf.read_last(4), [0, 1, 2, 3])


def test_starts_with_zeros():
    buf = RingBuffer(5)
    buf.write(np.array([7, 8]))
    np.testing.assert_array_equal(buf.read_last(5), [0, 0, 0, 7, 8])


def test_write_that_wraps_around_the_end():
    buf = RingBuffer(5)
    buf.write(np.arange(4))             # 0 1 2 3
    buf.write(np.arange(4, 7))          # 4 fits at the end, 5 and 6 wrap to the start
    np.testing.assert_array_equal(buf.read_last(5), [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(buf.read_last(2), [5, 6])


def test_many_blocks_match_the_whole_stream():
    rng = np.random.default_rng(0)
    buf = RingBuffer(100)
    stream = []
    for _ in range(50):                 # blocks of random length, many wraps
        block = rng.standard_normal(rng.integers(1, 40)).astype(np.float32)
        buf.write(block)
        stream.append(block)
        whole = np.concatenate(stream)
        m = min(len(whole), 100)
        np.testing.assert_array_equal(buf.read_last(m), whole[-m:])


def test_block_longer_than_the_buffer_keeps_its_end():
    buf = RingBuffer(5)
    buf.write(np.arange(12))
    np.testing.assert_array_equal(buf.read_last(5), [7, 8, 9, 10, 11])
    buf.write(np.array([12]))
    np.testing.assert_array_equal(buf.read_last(3), [10, 11, 12])


def test_read_returns_a_copy():
    buf = RingBuffer(5)
    buf.write(np.arange(5))
    out = buf.read_last(5)
    out[:] = -1                         # changing the result must not change the buffer
    np.testing.assert_array_equal(buf.read_last(5), [0, 1, 2, 3, 4])


def test_frame_levels_of_a_sine_and_silence():
    t = np.arange(1000) / 1000
    y = np.concatenate([np.sin(2 * np.pi * 50 * t), np.zeros(1000)])    # 1 s of sine, then silence
    levels = frame_levels_db(y, 100)
    assert len(levels) == 20
    np.testing.assert_allclose(levels[:10], 10 * np.log10(0.5), atol=1e-6)   # power of a sine = A^2/2
    assert np.all(levels[10:] < -100)


def test_comfort_noise_level():
    y = add_comfort_noise(np.zeros(44100, dtype=np.float32), -60, rng=0)
    assert y.dtype == np.float32
    np.testing.assert_allclose(10 * np.log10(np.mean(y ** 2)), -60, atol=0.1)


def test_packaged_model_classifies_a_window():
    models, classes = load_packaged_model(MODEL)
    assert len(classes) == 50 and classes[0] == "dog"
    y = 0.01 * np.random.default_rng(0).standard_normal(2 * 22050).astype(np.float32)   # 2 s of noise
    p = class_probabilities(models, y)
    assert p.shape == (50,)
    np.testing.assert_allclose(p.sum(), 1, atol=1e-5)
