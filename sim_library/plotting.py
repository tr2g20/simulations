import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from qutip import Bloch, Qobj, sigmax, sigmay, sigmaz, expect
from tabulate import tabulate


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

def plot_bloch(wave_func: np.ndarray):
    """
    Plots trajectory of single qubit state on the Bloch sphere.
    Colour-coded from blue to red (inital to final state)

    Raises:
        ValueError: If the input states are not single-qubit states (i.e.,
            the second dimension of the array is not size 2)

    Args:
        wave_func (np.ndarray): A NumPy array representing the time-dependent
            wave function. Expected shape is (N, 2), where N is the number of
            time steps and the second dimension contains the complex amplitudes
            of the qubit state.
    """
    states = []
    for i in range(len(wave_func)):
        states.append(Qobj(wave_func[i,:]))

    sx, sy, sz = sigmax(), sigmay(), sigmaz()

    x = [expect(sx, state) for state in states]
    y = [expect(sy, state) for state in states]
    z = [expect(sz, state) for state in states]


    steps = len(states)
    r = np.linspace(0, 1, steps)
    g = np.zeros(steps)
    b = np.linspace(1, 0, steps)
    colours = np.column_stack((r, g, b))

    b = Bloch()
    b.add_points([x, y, z], meth='m', colors=colours) 

    b.add_states(state=states[0], colors=['b'])
    b.add_states(state=states[-1], colors=['r'])

    b.show()

def display_table(basis: np.ndarray, wave_func: np.ndarray):
    
    indices = np.arange(0, len(basis))
    binary = []
    for i in range(len(basis)):
        binary.append(f'{np.round(wave_func[i], 3)}|{basis[i] % 8:03b}>')

    data = list(zip(indices, basis, binary, np.round(np.square(np.abs(wave_func)), 3)))

    titles = ['Index', 'Momentum (hbar*k)', 'Psi', '|Psi|^2']

    print(tabulate(data, headers=titles, tablefmt="grid", stralign="right"))

def display_table_compare(basis: np.ndarray, init_state: np.ndarray, final_state: np.ndarray):

    indices = np.arange(0, len(basis))
    binary_init = []
    binary_fin = []
    for i in range(len(basis)):
        binary_init.append(f'{np.round(init_state[i], 3)}|{basis[i] % 8:03b}>')
        binary_fin.append(f'{np.round(final_state[i], 3)}|{basis[i] % 8:03b}>')

    data = list(zip(indices, basis, binary_init, binary_fin, np.round(np.square(np.abs(init_state)), 3), np.round(np.square(np.abs(final_state)), 3)))

    titles = ['Index', 'Momentum (hbar*k)', 'Initial Psi', 'Final Psi', 'Initial |Psi|^2', 'Final |Psi|^2']

    print(tabulate(data, headers=titles, tablefmt="grid", stralign="right"))

def plot_coolingcycles_23(moms_list: list, fracs_list: list, basis: np.ndarray, cycles_list: np.ndarray, index_list: np.ndarray, bins_list: np.ndarray, temp: float = -1, show: bool = True):
    
    fig, axes = plt.subplots(3,2, figsize=(9, 7.5), layout= 'constrained')
    axes = axes.flatten()
    
    peak_density=0
    for i in range(0,6):
        counts, _, _ = axes[i].hist(x = moms_list[index_list[i]], weights= fracs_list[index_list[i]], bins = bins_list[i], histtype='step', density=True)
        axes[i].set_xticks(basis[0::2], basis[0::2], fontsize = 9)
        axes[i].set_xlim(basis[0],basis[-1])
        current = np.max(counts)
        if current > peak_density:
            peak_density = current

    yticks = np.arange(int(peak_density*1.1*10)+1)*0.1

    for i in range(0,6):
        axes[i].set_ylim(0,peak_density*1.1)
        axes[i].set_yticks(yticks)
        axes[i].text((len(basis)*0.81)+basis[0],peak_density*1.1*0.9,rf'{cycles_list[i]} Cycles')

    axes[4].set_xlabel(r'$p$ ($\hbar k_{eff}$)')
    axes[5].set_xlabel(r'$p$ ($\hbar k_{eff}$)')
    axes[0].set_ylabel('Probability Density')
    axes[2].set_ylabel('Probability Density')
    axes[4].set_ylabel('Probability Density')

    if temp > -1:
        axes[0].text((len(basis)*0.04)+basis[0],peak_density*1.1*0.9,rf'$T_{{init}}={temp*1e6:.0f}\mu K$')

    if show:
        fig.show()
