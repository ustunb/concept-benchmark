"""Generate synetic timeseries dataset."""
import numpy as np
import pandas as pd
from scipy import signal
from concept_benchmark.data import ConceptDataset

# Concepts constants
OSCILLATION_WAVEFORMS = ["sine", "square", "sawtooth"]
OSCILLATION_FREQ = [0, 5, 15, 30]
NOISE_COLORS = ["white", "pink", "red", "brownian"]
NOISE_LEVEL = {"low": 0.2, "medium": 1.0, "high": 3.0}
NOISE_COLOR_EXP = {"white": 0.0, "pink": 0.5, "red": 1.0, "brownian": 2.0}

def gen_noise(beta: float, n_samples: int, rng: int) -> np.ndarray:
    """
    Simulate colored noise.

    :param beta: slope of noise in spectral space
    :param n_samples: number of timepoints to generate
    :param rng: random number generator
    :returns y: time series
    """
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    S = np.zeros_like(freqs)
    mask = freqs > 0
    S[mask] = 1.0 / (freqs[mask]**beta)
    phases = rng.uniform(0, 2*np.pi, size=len(S))
    spectrum = S * np.exp(1j * phases)
    y = np.fft.irfft(spectrum, n=n_samples)
    return y / np.std(y)

def gen_waveform(waveform, freq, n_samples, fs) -> np.ndarray:
    """
    Generate osillatory waveform.

    :param waveform: oscillation wave, {"sine", "square", "sawtooth"}
    :param freq: frequency in Hertz, {None, 5, 15, 30}
    :param n_samples: number of samples
    :param fs: sampling rate, in Hertz
    """
    time = np.arange(n_samples) / fs
    if freq == 0:
        return np.zeros(n_samples)
    if waveform == "sine":
        return np.sin(2*np.pi*freq*time)
    if waveform == "square":
        return signal.square(2*np.pi*freq*time)
    if waveform == "sawtooth":
        return signal.sawtooth(2*np.pi*freq*time)
    raise ValueError(f"Unknown waveform: {waveform}")


def simulate_timeseries(stability_class, n_seconds=5, fs=1000, seed=None) -> tuple[np.ndarray, dict]:
    """
    Simluate a timeseries of a given class.

    :param stability_class: {
        0: stationary (stable)
        1: weakly non-stationary (slow drift, small bursts)
        2: strongly non-stationary (frequent or large bursts)
        3: unstable (variance explodes)
        }
    :param n_seconds: seconds to simulate
    :param fs: sampling rate, in Hertz
    :param seed: random seed
    :returns ConcepDataset:

    Classification Map
    ------------------

    Stability class depends on non-stationarity, long memory, and variance stability:
        stability_class == 0: nonstat=0, longmem=0, var=stable
        stability_class == 1: nonstat=1, longmem=0, var=stable
        stability_class == 2: nonstat=1, longmem=1, var=stable
        stability_class == 3: nonstat=1, longmem=1, var=unstable

    Non-stationarity depends on bursting state.
        nonstat == 1, bursting == 0
        nonstat == 0, bursting == 1

    Long-memory depends on noise color.
        longmem == 0: noise_color == "white"
        longmem == 1: noise_color in ("pink", "red", "brownian")

    Unstable class requires exploding variance:
        var == stable   : exponential_growth == 0
        var == unstable : exponential_growth == 1

    Other concepts:
        oscillation waveform and frequency
        noise standard deviation
    """

    rng = np.random.default_rng(seed)
    n_samples = fs * n_seconds

    # Non-stationarity and long memory
    nonstat = 0 if stability_class == 0 else 1
    longmem = 0 if stability_class < 2 else 1

    # Concepts
    waveform = rng.choice(OSCILLATION_WAVEFORMS)
    noise_color = rng.choice(["pink", "red", "brownian"] if longmem else ["white"])
    noise_level = rng.choice(list(NOISE_LEVEL.keys()))
    bursting = bool(nonstat)
    freq = rng.choice(OSCILLATION_FREQ[1:]) if bursting else OSCILLATION_FREQ[0]

    # Generate
    wave = gen_waveform(waveform, freq, n_samples, fs)
    noise = gen_noise(NOISE_COLOR_EXP[noise_color], n_samples, rng)
    x = wave + NOISE_LEVEL[noise_level]*noise

    if bursting:
        burst_mask = np.zeros(n_samples)
        burst_len = int(0.2*fs)
        for _ in range(rng.integers(1, 4)):
            start = rng.integers(0, n_samples-burst_len)
            burst_mask[start:start+burst_len] = 1
        x += burst_mask * rng.normal(0, 3*NOISE_LEVEL[noise_level], size=n_samples)

    if stability_class == 3:
        growth = np.exp(np.linspace(0, 3, n_samples)**2)
        x *= growth

    c = dict(
        waveform=waveform,
        freq=freq,
        noise_color=noise_color,
        noise_level=noise_level,
        bursting=bursting
    )

    return x, c

def simulate_timeseries_dataset(n, n_seconds, fs, seeds=None):

    d = int(n_seconds * fs)
    seeds = [None] * n if seeds is None else seeds
    y = np.random.choice(4, n)
    X = np.zeros((n, d))
    C = []
    for i in range(n):
        X[i], c = simulate_timeseries(y[i], fs, n_seconds)
        C.append(c)

    # Hot encode concepts
    df = pd.DataFrame(C)
    df["waveform"] = pd.Categorical(df['waveform'], categories=OSCILLATION_WAVEFORMS)
    df["freq"] = pd.Categorical(df['freq'], categories=OSCILLATION_FREQ)
    df["noise_color"] = pd.Categorical(df['noise_color'], categories=NOISE_COLORS)
    df["noise_level"] = pd.Categorical(df['noise_level'], categories=list(NOISE_LEVEL.keys()))
    C = pd.get_dummies(df)

    # To dataset object
    meta = dict(
        data_type="timeseries",
        classes=["stationary", "weakly non-stationary", "strongly non-stationary", "unstable"],
        concepts=list(C.columns),
    )

    return ConceptDataset(X, C.values, y, meta)
