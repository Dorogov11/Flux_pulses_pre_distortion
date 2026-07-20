import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def square_pulse(
    on_time: int, lpad: int, rpad: int, amp: int = 1, dt: float = 1e-9
) -> Tuple[np.ndarray, List[float]]:
    total_len = lpad + rpad + on_time
    pulse = ([0] * lpad) + ([amp] * on_time) + ([0] * rpad)
    timesteps = np.array([dt * i for i in range(total_len)])
    return timesteps, pulse


def gaussian_pulse(sigma, chop, lpad, rpad, amp=1, dt=1e-9) -> tuple[np.ndarray]:
    """ """
    length = int(sigma * chop)
    total_len = lpad + length + rpad

    start, stop = -chop / 2 * sigma, chop / 2 * sigma
    ts = np.arange(start, stop, 1)
    exponential = np.exp(-(ts**2) / (2.0 * sigma**2))
    pulse = amp * exponential
    pulse = np.concatenate(([0] * lpad, pulse, [0] * rpad))

    timesteps = np.array([dt * i for i in range(total_len)])
    return timesteps, pulse


# generates a generic square pulse that lets users determine the starting, hold and end amps
def square_generic(
    on_time: int,
    lpad: int,
    rpad: int,
    amp: Tuple[float, float, float],
    dt: float = 1e-9,
) -> Tuple[np.ndarray, List[float]]:
    total_len = lpad + rpad + on_time
    pulse = ([amp[0]] * lpad) + ([amp[1]] * on_time) + ([amp[2]] * rpad)
    timesteps = np.array([dt * i for i in range(total_len)])
    return timesteps, pulse


def constant_cosine(length_constant, length_ring, lpad=0, rpad=0, amp=1, dt=1e-9):
    def ring_up_wave(length_ring, reverse=False, shape="cos"):
        if shape == "cos":
            i_wave = ring_up_cos(length_ring)
        elif shape == "tanh":
            i_wave = ring_up_tanh(length_ring)
        else:
            raise ValueError("Type must be 'cos' or 'tanh', not %s" % shape)
        if reverse:
            i_wave = i_wave[::-1]
        return i_wave

    def ring_up_cos(length_ring):
        return 0.5 * (1 - np.cos(np.linspace(0, np.pi, length_ring))) * amp

    def ring_up_tanh(length_ring):
        ts = np.linspace(-2, 2, length_ring)
        return (1 + np.tanh(ts)) / 2 * amp

    ring_up = ring_up_wave(length_ring)
    ring_down = ring_up_wave(length_ring, reverse=True)
    constant = np.full(length_constant, amp)
    pulse = np.concatenate(([0] * lpad, ring_up, constant, ring_down, [0] * rpad))
    # pulse = [0] * lpad + list(pulse)
    total_length = lpad + length_constant + 2 * length_ring + rpad

    timesteps = np.array([dt * i for i in range(total_length)])
    return timesteps, pulse


def constant_cosine_reset(length_constant, length_ring, lpad=0, rpad=0, amp=1, dt=1e-9):
    def ring_up_wave(length_ring, reverse=False, shape="cos"):
        if shape == "cos":
            i_wave = ring_up_cos(length_ring)
        elif shape == "tanh":
            i_wave = ring_up_tanh(length_ring)
        else:
            raise ValueError("Type must be 'cos' or 'tanh', not %s" % shape)
        if reverse:
            i_wave = i_wave[::-1]
        return i_wave

    def ring_up_cos(length_ring):
        return 0.5 * (1 - np.cos(np.linspace(0, np.pi, length_ring))) * amp

    def ring_up_tanh(length_ring):
        ts = np.linspace(-2, 2, length_ring)
        return (1 + np.tanh(ts)) / 2 * amp

    ring_up = ring_up_wave(length_ring)
    ring_middle = 2 * ring_up_wave(2 * length_ring, reverse=True) - 1
    ring_down = ring_up_wave(length_ring, reverse=False) - 1
    constant = np.full(length_constant, amp)
    constant_reverse = np.full(length_constant, -amp)
    pulse = np.concatenate(
        (
            [0] * lpad,
            ring_up,
            constant,
            ring_middle,
            constant_reverse,
            ring_down,
            [0] * rpad,
        )
    )
    # pulse = [0] * lpad + list(pulse)
    total_length = lpad + length_constant * 2 + 4 * length_ring + rpad

    timesteps = np.array([dt * i for i in range(total_length)])
    return timesteps, pulse

def double_square(
    t1: int,
    t2: int,
    lpad: int,
    rpad: int,
    A: int,
    B: int,
    dt: float = 1e-9,
):

    pulse = (
        [0] * lpad
        + [A] * t1
        + [B] * t2
        + [A] * t1
        + [0] * rpad
    )

    timesteps = np.arange(len(pulse)) * dt

    return timesteps, pulse

def double_square_ramped(
    t1: int,
    t2: int,
    outer_ramp: int,
    inner_ramp: int,
    lpad: int = 0,
    rpad: int = 0,
    A: float = 1.0,
    B: float = 0.4,
    dt: float = 1e-9,
):

    def ramp(start, stop, length):
        return start + (stop - start) * 0.5 * (1 - np.cos(np.linspace(0, np.pi, length)))

    pulse = np.concatenate((
        np.zeros(lpad),
        ramp(0, A, outer_ramp),
        np.full(t1, A),
        ramp(A, B, inner_ramp),
        np.full(t2, B),
        ramp(B, A, inner_ramp),
        np.full(t1, A),
        ramp(A, 0, outer_ramp),
        np.zeros(rpad),
    ))

    return np.arange(len(pulse)) * dt, pulse

def plot_pulse(timesteps, pulse, title="Pulse"):
    """
    Plot a pulse waveform.
    """
    plt.figure(figsize=(8, 3))
    plt.plot(timesteps, pulse, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


PLOT_PULSE = True

if __name__ == "__main__" and PLOT_PULSE:

    ts, pulse = double_square_ramped(
        t1=2000,
        t2=5000,
        outer_ramp=400,
        inner_ramp=200,
        lpad=800,
        rpad=1000,
        A=1,
        B=0.4,
    )

    plot_pulse(ts, pulse, title="Double Square Pulse")
# import matplotlib.pyplot as plt

# ts, pulse = gaussian_pulse(
#     sigma=500,
#     chop=8,
#     lpad=200,
#     rpad=200,
#     dt=1e-9,
#     amp=1,
# )
# plt.scatter(ts, pulse)
# plt.show()
