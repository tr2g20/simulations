import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation


def plot_state_trajectories(ax: Axes, wavefunc: np.ndarray, time_array: np.ndarray, basis: np.ndarray, title: str = 'State trajectories', contains_title: bool = True):
    """
    Plots the state trajectories against times against time for a pulse sequence.

    Args:
        ax: The Axes object to draw the histograms onto.
        mom_vals: The 2D array of the state vector as a function of time.
        time_array: The 1D array of the time steps.
        basis: The 1D array of momentum basis states (in integers of hbar*k_eff).
        title: The title for the plot.

    """
    ax.plot(time_array*1e-6, wavefunc)
    ax.legend(basis, loc='upper left')
    ax.set_ylabel('Probability Amplitude')
    ax.set_xlabel(r'Time ($\mu s$)')
    if contains_title:
        ax.set_title(f'{title}')

def plot_hist(ax: Axes, mom_vals: np.ndarray, fracs: np.ndarray, n_bins: int, n_atoms: int, temp: float, title: str = 'Momentum Distribution', contains_title: bool = True):
    """
    Plots the initial and final momentum distribution on one histogram. 
    Modifies the Axes object that is inputted.

    Args:
        ax: The Axes object to draw the histograms onto.
        mom_vals: A 1D array representing the possible momentum values.
        fracs: A 2D array of corresponding state fractions at every time step.
        n_bins: Number of bins for the histogram.
        n_atoms: Total number of atoms.
        temp: The temperature of the cloud in Kelvin.
        title: The base title for the plot.

    """
    ax.hist(x= mom_vals, weights= fracs[0], bins = n_bins, histtype='step', density=True, label= 'Initial')
    ax.hist(x= mom_vals, weights= fracs[-1], bins = n_bins, histtype='step', density=True, label= 'Final')
    if contains_title:
        ax.set_title(rf"{title} ($N={n_atoms}$, $T={temp*1e6:.3g}\mu K$)")
    ax.set_ylabel('Probability Density')
    ax.set_xlabel(r"Momentum ($\hbar k_{eff}$)")
    ax.legend()

def save_gif(filename: str, mom_vals: np.ndarray, fracs: np.ndarray, n_bins: int, wavefunc: np.ndarray, times: np.ndarray, basis: np.ndarray, frame_interval: float, pause_time: float, mom_ylim: float):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7.5), layout= 'constrained')

    def update(frame):
        ax1.clear() 
        ax2.clear()
        ax1.plot(times[:frame]*1e6, wavefunc[:frame])
        ax1.set_ylim(0,1)
        ax1.set_xlim(0, times[-1]*1e6)
        ax1.legend(basis, loc='upper left')
        ax1.set_ylabel('Probability Amplitude')
        ax1.set_xlabel(r'Time ($\mu s$)')
        ax1.set_title(rf'State amplitudes at time {times[frame]*1e6:.1f}$\mu$s')
        ax2.set_ylim(0, mom_ylim)
        ax2.set_xlabel(r"Momentum ($\hbar k_{eff}$)")
        ax2.set_ylabel("Probability Density")
        ax2.set_title(rf"Momentum distribution at time {times[frame]*1e6:.1f}$\mu$")
        ax2.hist(x= mom_vals, weights= fracs[frame], bins = n_bins, histtype='step', density=True)

    n_frames = len(fracs)
    pause_frames = round(pause_time/frame_interval)
    frame_indices = list(range(n_frames)) + [n_frames - 1] * pause_frames

    print(f'{n_frames} frames')

    ani = FuncAnimation(fig, update, frames=frame_indices, interval= frame_interval)
    ani.save(filename, writer='pillow')