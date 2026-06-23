import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from qutip import Bloch, Qobj, sigmax, sigmay, sigmaz, expect
from tabulate import tabulate
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sim_library.constants import kb, m, pi, hbar, k_eff


def plot_state_trajectories(ax: Axes, wavefunc: np.ndarray, time_array: np.ndarray, basis: np.ndarray, title: str = 'State trajectories', contains_title: bool = True):
    """
    Plots the probability amplitudes of each state against time for a pulse sequence.

    Args:
        ax: The Axes object to draw the trajectories onto.
        wavefunc: The 2D array of the state vector as a function of time.
        time_array: The 1D array of the time steps.
        basis: The 1D array of momentum basis states (in integers of hbar*k_eff).
        title: The title for the plot.
        contains_title: If True, adds the title to the plot.
    """
    amplitudes = np.abs(wavefunc)**2

    ax.plot(time_array*1e6, amplitudes)
    ax.legend(basis, loc='upper left')
    ax.set_ylabel('Probability Amplitude')
    ax.set_xlabel(r'Time ($\mu s$)')
    if contains_title:
        ax.set_title(f'{title}')

def plot_hist(ax: Axes, mom_vals: np.ndarray, fracs: np.ndarray, n_bins: int, basis:np.ndarray, n_atoms: int, temp: float, title: str = 'Momentum Distribution', contains_title: bool = True):
    """
    Plots the initial and final momentum distribution on one histogram. 
    Modifies the Axes object that is inputted.

    Args:
        ax (Axes): The Axes object to draw the histograms onto.
        mom_vals (np.ndarray): A 1D array representing the possible momentum values.
        fracs (np.ndarray): A 2D array of corresponding state fractions at every time step.
        n_bins (int): Number of bins for the histogram.
        basis (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        n_atoms (int): Total number of atoms.
        temp (float): The temperature of the cloud in Kelvin.
        title (str): The base title for the plot.
        contains_title: If True, adds the title to the plot.
    """
    # ax.hist(x= mom_vals, weights= fracs[0], bins = n_bins, histtype='step', density=True, label= 'Initial')
    # ax.hist(x= mom_vals, weights= fracs[-1], bins = n_bins, histtype='step', density=True, label= 'Final')
    labels = ['Initial', 'Final']

    for i in [0,-1]:
        counts, bin_edges = np.histogram(mom_vals, weights=fracs[i], bins = n_bins, density=True)
        bindiff = bin_edges[1]-bin_edges[0]
        binmids = bin_edges[:-1] + (bindiff/2)
        ax.plot(binmids, counts, linewidth=1, label=labels[i])
        ax.set_xticks(basis[0::2], basis[0::2], fontsize = 9)
        ax.set_xlim(basis[0],basis[-1])

    if contains_title:
        ax.set_title(rf"{title} ($N={n_atoms}$, $T={temp*1e6:.3g}\mu K$)")
    ax.set_ylabel('Probability Density')
    ax.set_xlabel(r"Momentum ($\hbar k_{eff}$)")
    ax.legend()

def save_gif(filename: str, mom_vals: np.ndarray, fracs: np.ndarray, n_bins: int, wavefunc: np.ndarray, times: np.ndarray, basis: np.ndarray, frame_interval: float, pause_time: float, mom_ylim: float):
    """
    Generates and saves an animated GIF of state trajectories and momentum distributions over time.
    Each time step is used as a frame. For a gif with equal time steps use save_gif_interp.

    Args:
        filename (str): The file path and name to save the output GIF.
        mom_vals (np.ndarray): A 1D array representing the possible momentum values.
        fracs (np.ndarray): A 2D array of corresponding state fractions at every time step.
        n_bins (int): The number of bins to use for the momentum histogram.
        wavefunc (np.ndarray): 2D array of the state amplitudes (mod squared) plotted against time.
        times (np.ndarray): 1D array of time steps in seconds.
        basis (np.ndarray): 1D array of momentum basis states (in integers of hbar*k_eff).
        frame_interval (float): Delay between frames in milliseconds.
        pause_time (float): The total duration in milliseconds to hold the final frame.
        mom_ylim (float): The upper limit for the y-axis of the momentum distribution plot.
    """
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

def save_gif_interp(filename: str, mom_vals: np.ndarray, fracs: np.ndarray, n_bins: int, wavefunc: np.ndarray, times: np.ndarray, basis: np.ndarray, n_frames: int, frame_interval: float, pause_time: float, mom_ylim: float):
    """
    Generates and saves an animated GIF of state trajectories and momentum distributions over time.
    The time steps of fracs and wavefunc arrays are adjusted to be equally spaced through linear interpolation. 
    This ensures the time counter increases at a uniform rate (this doesnt work well at high Rabi frequencies 
    as pulses are too short compared to freevolution).

    Args:
        filename (str): The file path and name to save the output GIF.
        mom_vals (np.ndarray): A 1D array representing the possible momentum values.
        fracs (np.ndarray): A 2D array of corresponding state fractions at every time step.
        n_bins (int): The number of bins to use for the momentum histogram.
        wavefunc (np.ndarray): 2D array of the state amplitudes (mod squared) plotted against time.
        times (np.ndarray): 1D array of original time steps in seconds.
        basis (np.ndarray): 1D array of momentum basis states (in integers of hbar*k_eff).
        n_frames (int): The number of linearly spaced frames to interpolate the animation over.
        frame_interval (float): Delay between frames in milliseconds.
        pause_time (float): The total duration in milliseconds to hold the final frame.
        mom_ylim (float): The upper limit for the y-axis of the momentum distribution plot.
    """
    linear_times = np.linspace(times[0], times[-1], n_frames)

    wavefunc_interpolator = interp1d(times, wavefunc, axis=0, kind='linear')
    fracs_interpolator = interp1d(times, fracs, axis=0, kind='linear')
    times = linear_times

    wavefunc = wavefunc_interpolator(times)
    fracs = fracs_interpolator(times)
    
    
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

def plot_bloch(wave_func: np.ndarray, labels: list[str] = ['$\\left| 0\\right\\rangle$', '$\\left| 1\\right\\rangle$'], show=True):
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
        labels (list[str]): Labels for north and south poles of Bloch sphere.
            Default is |0> and |1>.
        show (bool): If true runs fig.show() command for Bloch object.
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

    b.zlabel = labels

    if show:
        b.show()  
          
    return b

def display_table(basis: np.ndarray, wave_func: np.ndarray):
    """
    Prints a formatted table displaying the index, momentum/binary representation, complex amplitude and
    mod squared amplitude of each pure state.

    Args:
        basis (np.ndarray): 1D array of momentum basis states (in integers of hbar*k_eff).
        wave_func (np.ndarray): 1D array of the complex state amplitudes.
    """

    indices = np.arange(0, len(basis))
    binary = []
    for i in range(len(basis)):
        binary.append(f'{np.round(wave_func[i], 3)}|{basis[i] % 8:03b}>')

    data = list(zip(indices, basis, binary, np.round(np.square(np.abs(wave_func)), 3)))

    titles = ['Index', 'Momentum (hbar*k)', 'Psi', '|Psi|^2']

    print(tabulate(data, headers=titles, tablefmt="grid", stralign="right"))

def display_table_compare(basis: np.ndarray, init_state: np.ndarray, final_state: np.ndarray):
    """
    Prints a formatted table displaying the index, momentum/binary representation, complex amplitude and
    mod squared amplitude of each pure state for two wavefunctions for comparison

    Args:
        basis (np.ndarray): 1D array of momentum basis states (in integers of hbar*k).
        init_state (np.ndarray): 1D array of the initial complex state amplitudes.
        final_state (np.ndarray): 1D array of the final complex state amplitudes.
    """

    indices = np.arange(0, len(basis))
    binary_init = []
    binary_fin = []
    for i in range(len(basis)):
        binary_init.append(f'{np.round(init_state[i], 3)}|{basis[i] % 8:03b}>')
        binary_fin.append(f'{np.round(final_state[i], 3)}|{basis[i] % 8:03b}>')

    data = list(zip(indices, basis, binary_init, binary_fin, np.round(np.square(np.abs(init_state)), 3), np.round(np.square(np.abs(final_state)), 3)))

    titles = ['Index', 'Momentum (hbar*k)', 'Initial Psi', 'Final Psi', 'Initial |Psi|^2', 'Final |Psi|^2']

    print(tabulate(data, headers=titles, tablefmt="grid", stralign="right"))

def plot_coolingcycles_21(moms_list: list, fracs_list: list, basis: np.ndarray, datarange: tuple, cycles_list: np.ndarray, index_list: np.ndarray, bins_list: np.ndarray, title: str = '', ylim: float = 0, autoyticks: bool = False, temp: float = -1, show: bool = True, save_dir: str = ''):
    """
    Plots momentum distribution histograms in 1x2 panels for different numbers of cooling cycles.

    Args:
        moms_list (list): List of 1D arrays representing momentum distributions for each simulation step.
        fracs_list (list): List of 1D arrays containing corresponding state fractions/weights.
        basis (np.ndarray): 1D array of momentum basis states (in integers of hbar*k_eff).
        datarange (tuple): The global (min, max) momentum values for generating the histograms.
        cycles_list (np.ndarray): List of cycle numbers to label each subplot.
        index_list (np.ndarray): List of indices of moms_list/fracs_list to plot.
        bins_list (np.ndarray): List of number of histogram bins for each subplot.
        title (str): The overall figure title.
        ylim (float): Manual upper limit for the y-axis. If 0, scales automatically to 1.1x the peak density.
        autoyticks (bool): If True, relies on Matplotlib for y-ticks. If False, manually calculates 0.1 increments.
        temp (float): Initial temperature in Kelvin for labeling. Ignored if <= -1.
        show (bool): If True, displays the figure.
        save_dir (str): File path to save the generated plot. If empty, saving is skipped.
    """

    fig, axes = plt.subplots(1,2, figsize=(9, 2.5), layout= 'constrained')
    axes = axes.flatten()

    peak_density=0
    for i in range(0,2):
        counts, bin_edges = np.histogram(moms_list[index_list[i]], weights=fracs_list[index_list[i]], bins = bins_list[i], density=True, range=datarange)
        bindiff = bin_edges[1]-bin_edges[0]
        binmids = bin_edges[:-1] + (bindiff/2)
        axes[i].plot(binmids, counts, linewidth=1, c = 'royalblue')
        axes[i].set_xticks(basis[0::2], basis[0::2], fontsize = 9)
        axes[i].set_xlim(basis[0],basis[-1])
        current = np.max(counts)
        if current > peak_density:
            peak_density = current

    if ylim == 0:
        top = peak_density*1.1
    else:
        top = ylim
    if not autoyticks:
        yticks = np.arange(int(top*10)+1)*0.1

    for i in range(0,2):
        axes[i].set_ylim(0,top)
        if not autoyticks:
            axes[i].set_yticks(yticks)
        axes[i].text((len(basis)*0.81)+basis[0],top*0.9,rf'{cycles_list[i]} Cycles')

    axes[0].set_xlabel(r'$p$ ($\hbar k_{eff}$)')
    axes[0].set_xlabel(r'$p$ ($\hbar k_{eff}$)')
    axes[0].set_ylabel('Probability Density')
    axes[1].set_ylabel('Probability Density')

    if temp > -1:
        axes[0].text((len(basis)*0.04)+basis[0],top*0.9,rf'$T_{{init}}={temp*1e6:.0f}\mu K$')

    if title != '':
        fig.suptitle(title)

    if show:
        fig.show()

    if save_dir != '':
        if not Path(save_dir).exists():
            fig.savefig(fname = save_dir, dpi=1000, bbox_inches='tight')
        else:
            print(f"'{save_dir}' already exists. Save aborted.")

def plot_coolingcycles_22(moms_list: list, fracs_list: list, basis: np.ndarray, datarange: tuple, cycles_list: np.ndarray, index_list: np.ndarray, bins_list: np.ndarray, title: str ='', ylim: float = 0, autoyticks: bool = False, temp: float = -1, show: bool = True, save_dir: str = ''):
    """
    Plots momentum distribution histograms in 2x2 panels for different numbers of cooling cycles.

    Args:
        moms_list (list): List of 1D arrays representing momentum distributions for each simulation step.
        fracs_list (list): List of 1D arrays containing corresponding state fractions/weights.
        basis (np.ndarray): 1D array of momentum basis states (in integers of hbar*k_eff).
        datarange (tuple): The global (min, max) momentum values for generating the histograms.
        cycles_list (np.ndarray): List of cycle numbers to label each subplot.
        index_list (np.ndarray): List of indices of moms_list/fracs_list to plot.
        bins_list (np.ndarray): List of number of histogram bins for each subplot.
        title (str): The overall figure title.
        ylim (float): Manual upper limit for the y-axis. If 0, scales automatically to 1.1x the peak density.
        autoyticks (bool): If True, relies on Matplotlib for y-ticks. If False, manually calculates 0.1 increments.
        temp (float): Initial temperature in Kelvin for labeling. Ignored if <= -1.
        show (bool): If True, displays the figure.
        save_dir (str): File path to save the generated plot. If empty, saving is skipped.
    """

    fig, axes = plt.subplots(2,2, figsize=(9, 5), layout= 'constrained')
    axes = axes.flatten()
    
    peak_density=0
    for i in range(0,4):
        counts, bin_edges = np.histogram(moms_list[index_list[i]], weights=fracs_list[index_list[i]], bins = bins_list[i], density=True, range=datarange)
        bindiff = bin_edges[1]-bin_edges[0]
        binmids = bin_edges[:-1] + (bindiff/2)
        axes[i].plot(binmids, counts, linewidth=1, c = 'royalblue')
        axes[i].set_xticks(basis[0::2], basis[0::2], fontsize = 9)
        axes[i].set_xlim(basis[0],basis[-1])
        current = np.max(counts)
        if current > peak_density:
            peak_density = current

    if ylim == 0:
        top = peak_density*1.1
    else:
        top = ylim

    if not autoyticks:
        yticks = np.arange(int(top*10)+1)*0.1

    for i in range(0,4):
        axes[i].set_ylim(0,top)
        if not autoyticks:
            axes[i].set_yticks(yticks)
        axes[i].text((len(basis)*0.81)+basis[0],top*0.9,rf'{cycles_list[i]} Cycles')

    axes[2].set_xlabel(r'$p$ ($\hbar k_{eff}$)')
    axes[3].set_xlabel(r'$p$ ($\hbar k_{eff}$)')
    axes[0].set_ylabel('Probability Density')
    axes[2].set_ylabel('Probability Density')

    if temp > -1:
        axes[0].text((len(basis)*0.04)+basis[0],top*0.9,rf'$T_{{init}}={temp*1e6:.0f}\mu K$')

    if title != '':
        fig.suptitle(title)

    if show:
        fig.show()

    if save_dir != '':
        if not Path(save_dir).exists():
            fig.savefig(fname = save_dir, dpi=1000, bbox_inches='tight')
        else:
            print(f"'{save_dir}' already exists. Save aborted.")
        

def plot_coolingcycles_23(moms_list: list, fracs_list: list, basis: np.ndarray, datarange: tuple, cycles_list: np.ndarray, index_list: np.ndarray, bins_list: np.ndarray, title: str = '', ylim: float = 0, autoyticks: bool = False, temp: float = -1, show: bool = True, save_dir: str = ''):
    """
    Plots momentum distribution histograms in 3x2 panels for different numbers of cooling cycles.

    Args:
        moms_list (list): List of 1D arrays representing momentum distributions for each simulation step.
        fracs_list (list): List of 1D arrays containing corresponding state fractions/weights.
        basis (np.ndarray): 1D array of momentum basis states (in integers of hbar*k_eff).
        datarange (tuple): The global (min, max) momentum values for generating the histograms.
        cycles_list (np.ndarray): List of cycle numbers to label each subplot.
        index_list (np.ndarray): List of indices of moms_list/fracs_list to plot.
        bins_list (np.ndarray): List of number of histogram bins for each subplot.
        title (str): The overall figure title.
        ylim (float): Manual upper limit for the y-axis. If 0, scales automatically to 1.1x the peak density.
        autoyticks (bool): If True, relies on Matplotlib for y-ticks. If False, manually calculates 0.1 increments.
        temp (float): Initial temperature in Kelvin for labeling. Ignored if <= -1.
        show (bool): If True, displays the figure.
        save_dir (str): File path to save the generated plot. If empty, saving is skipped.
    """

    fig, axes = plt.subplots(3,2, figsize=(9, 7.5), layout= 'constrained')
    axes = axes.flatten()

    peak_density=0
    for i in range(0,6):
        counts, bin_edges = np.histogram(moms_list[index_list[i]], weights=fracs_list[index_list[i]], bins = bins_list[i], density=True, range=datarange)
        bindiff = bin_edges[1]-bin_edges[0]
        binmids = bin_edges[:-1] + (bindiff/2)
        axes[i].plot(binmids, counts, linewidth=1, c = 'royalblue')
        axes[i].set_xticks(basis[0::2], basis[0::2], fontsize = 9)
        axes[i].set_xlim(basis[0],basis[-1])
        current = np.max(counts)
        if current > peak_density:
            peak_density = current

    if ylim == 0:
        top = peak_density*1.1
    else:
        top = ylim

    if not autoyticks:
        yticks = np.arange(int(top*10)+1)*0.1

    for i in range(0,6):
        axes[i].set_ylim(0,top)
        if not autoyticks:
            axes[i].set_yticks(yticks)
        axes[i].text((len(basis)*0.81)+basis[0],top*0.9,rf'{cycles_list[i]} Cycles')

    axes[4].set_xlabel(r'$p$ ($\hbar k_{eff}$)')
    axes[5].set_xlabel(r'$p$ ($\hbar k_{eff}$)')
    axes[0].set_ylabel('Probability Density')
    axes[2].set_ylabel('Probability Density')
    axes[4].set_ylabel('Probability Density')

    if temp > -1:
        axes[0].text((len(basis)*0.04)+basis[0],top*0.9,rf'$T_{{init}}={temp*1e6:.0f}\mu K$')

    if title != '':
        fig.suptitle(title)

    if show:
        fig.show()

    if save_dir != '':
        if not Path(save_dir).exists():
            fig.savefig(fname = save_dir, dpi=1000, bbox_inches='tight')
        else:
            print(f"'{save_dir}' already exists. Save aborted.")

def gaussian(p, amp, T, mu):
    '''
    Gaussian that represents a momentum distribution of temperature T.
    p is in units of hbar*k.
    '''
    return (amp*hbar*k_eff/(np.sqrt(2*pi*m*kb*T)))*np.exp(-((hbar*k_eff)*(p-mu))**2/(2*kb*T*m))

def fit_gaussian(moms: np.ndarray, fracs: np.ndarray, range: tuple, nbins: int, threshold: float = 0.5, xscale: float = 1, plot: bool = False, print_vals = True, fit_npoints: int = 1000, init_guess: list = [1.0, 1e-6, 0.0], joindata: bool = False):
    """
    Fits a Gaussian curve to the highest peak in a momentum distribution.

    Args:
        moms (np.ndarray): 1D array of momentum values.
        fracs (np.ndarray): 1D array of corresponding state fractions/weights.
        range (tuple): The (min, max) momentum values for generating the histogram.
        nbins (int): The number of histogram bins.
        threshold (float): The fraction of the peak density used to dynamically determine the truncation limits for the fit. Default is 0.5 (i.e. FWHM)
        xscale (float): Scales the x-axis domain, centered on the fitted peak. Scaling is relative to the width of the truncated region. 
            Default is 1 which plots purely the truncated region plus a half bin width buffer on either side.
        plot (bool): If True, plots the truncated histogram data alongside the fitted curve.
        print_vals (bool): If True, prints the fitted parameters and their standard errors.
        fit_npoints (int): The number of data points used to plot the fitted line in the plot.
        init_guess (list): Initial guess for the Gaussian parameters [amplitude, temperature, shift].
        joindata (bool): If True, connects the plotted data points with a line.

    Returns:
        tuple: A tuple containing four elements:
            - amp (float): The fitted amplitude of the Gaussian.
            - T (float): The fitted width/temperature parameter.
            - shift (float): The fitted center shift of the distribution.
            - errs (np.ndarray): The 1D array of standard errors for the fitted parameters.
    """

    counts, bin_edges = np.histogram(moms, weights=fracs, bins = nbins, range=range)
    
    bindiff = bin_edges[1]-bin_edges[0]
    binmids = bin_edges[:-1] + (bindiff/2)

    # Convert to density
    norm_factor = np.sum(counts) * bindiff
    prob_density = counts / norm_factor

    # Poisson error (sigma = root(N))
    dataerr = np.sqrt(counts) / norm_factor

    # Find indices around highest peak
    peak_idx = np.argmax(prob_density)
    valid_idx = np.where(prob_density >= threshold*prob_density[peak_idx])[0]
    discontinuities = np.where(np.diff(valid_idx) > 1)[0] + 1 # Look where index jumps more than 1
    split_data = np.split(valid_idx, discontinuities)
    for block in split_data: # Find split block that contains the index of the peak
        if peak_idx in block:
            trunc_mask = block
            break
    
    # Truncate
    binmids_trunc = binmids[trunc_mask]
    prob_density_trunc = prob_density[trunc_mask]
    dataerr_trunc = dataerr[trunc_mask]

    # Bounds for finding optimal params
    lower_bounds = [0, 0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf]

    # Assigns empty bins an error equal to 1 count to stop divide by zero in curve_fit
    safe_dataerr = np.where(dataerr_trunc == 0, 1.0 / norm_factor, dataerr_trunc)

    fitted_vals, cov = curve_fit(gaussian, binmids_trunc, prob_density_trunc, sigma=safe_dataerr, absolute_sigma=True, bounds=(lower_bounds, upper_bounds), p0=init_guess)

    amp = fitted_vals[0]
    T = fitted_vals[1]
    shift = fitted_vals[2]
    errs = np.sqrt(np.diag(cov))

    if plot:
        fig, ax = plt.subplots(dpi=100)

        if xscale <= 1:
            x_min = binmids_trunc[0] - (0.5*bindiff)
            x_max = binmids_trunc[-1] + (0.5*bindiff)
        else:
            # Determine distance from peak to edge of FWHM
            left_dist = abs(shift - binmids_trunc[0])
            right_dist = abs(binmids_trunc[-1] - shift)
            max_dist = max(left_dist, right_dist)
            # Scale domain while keeping peak central
            x_min = shift - (max_dist * xscale)
            x_max = shift + (max_dist * xscale)

        fit_x = np.linspace(x_min, x_max, fit_npoints)
        fit_y = gaussian(fit_x, amp, T, shift)

        if joindata:
            data = ax.errorbar(binmids_trunc, prob_density_trunc, yerr=dataerr_trunc, c='rebeccapurple', marker='.', ls='-', capsize=3, label=r'$p$ dist.', zorder=1000)
            ax.errorbar(binmids, prob_density, yerr=dataerr, alpha = 0.3, c='rebeccapurple', marker='.', ls='-', capsize=3, label=r'$p$ dist.', zorder=999)
        else:
            data = ax.errorbar(binmids_trunc, prob_density_trunc, yerr=dataerr_trunc, c='rebeccapurple', marker='.', ls='none', capsize=3, label=r'$p$ dist.', zorder=1000)
            ax.errorbar(binmids, prob_density, yerr=dataerr, alpha = 0.3, c='rebeccapurple', marker='.', ls='none', capsize=3, label=r'$p$ dist.', zorder=999)
        
        fit, = ax.plot(fit_x, fit_y, c='orange', label='Gaussian fit')

        # Monte carlo sample 1000 curves within 1sigma of optimal params
        sample_params = np.random.multivariate_normal([amp, T, shift], cov, 1000)
        sample_fits = np.array([gaussian(fit_x, *params) for params in sample_params])
        
        # 1-sigma boundaries (15.9th and 84.1st percentiles)
        fiterr_low = np.percentile(sample_fits, 15.9, axis=0)
        fiterr_high = np.percentile(sample_fits, 84.1, axis=0)
        
        # Shade the error region 
        fill = ax.fill_between(fit_x, fiterr_low, fiterr_high, color='orange', alpha=0.3, label=r'$1\sigma$ band')

        ax.grid()
        ax.set_xlim(x_min,x_max)
        ax.set_xlabel(r'$p$ $(\hbar k_{eff})$')
        ax.set_ylabel(r'Probability density')
        ax.legend(handles=[data,fit,fill],labels=[r'$p$ dist.', 'Gaussian fit', r'$1\sigma$ band'])
        plt.show()
    
    if print_vals:
        print(f'Efficiency: {amp*100:.5g} +/- {errs[0]:.5g} %')
        print(f'Temperature: {T*1e6:.5g} +/- {errs[1]*1e6:.5g} muK')
        print(f'Center: {shift:.5g} +/- {errs[2]:.5g} hbark')

    return amp, T, shift, errs

def fit_gaussian_custom(moms: np.ndarray, fracs: np.ndarray, range: tuple, nbins: int, trunc_lims: tuple, xscale: float = 1, plot: bool = False, print_vals = True, fit_npoints: int = 1000, init_guess: list = [1.0, 1e-6, 0.0], joindata: bool = False):
    """
    Fits a Gaussian curve to a specified region of a momentum distribution histogram.

    Args:
        moms (np.ndarray): 1D array of momentum values.
        fracs (np.ndarray): 1D array of corresponding state fractions/weights.
        range (tuple): The (min, max) momentum values for generating the histogram.
        nbins (int): The number of histogram bins.
        trunc_lims (tuple): The (lower, upper) momentum limits for truncating the data before fitting.
        xscale (float): Scales the x-axis domain, centered on the fitted peak. Scaling is relative to the width of the truncated region. 
            Default is 1 which plots purely the truncated region.
        plot (bool): If True, plots the truncated histogram data alongside the fitted curve.
        print_vals (bool): If True, prints the fitted parameters and their standard errors.
        fit_npoints (int): The number of data points used to plot the fitted line in the plot.
        init_guess (list): Initial guess for the Gaussian parameters [amplitude, temperature, shift].
        joindata (bool): If True, connects the plotted data points with a line.

    Returns:
        tuple: A tuple containing four elements:
            - amp (float): The fitted amplitude of the Gaussian.
            - T (float): The fitted width/temperature parameter.
            - shift (float): The fitted center shift of the distribution.
            - errs (np.ndarray): The 1D array of standard errors for the fitted parameters.
    """

    counts, bin_edges = np.histogram(moms, weights=fracs, bins = nbins, range=range)
    
    bindiff = bin_edges[1]-bin_edges[0]
    binmids = bin_edges[:-1] + (bindiff/2)

    # Convert to density
    norm_factor = np.sum(counts) * bindiff
    prob_density = counts / norm_factor

    # Poisson error (sigma = root(N))
    dataerr = np.sqrt(counts) / norm_factor

    # Truncate
    trunc_mask = (binmids >= trunc_lims[0]) & (binmids <= trunc_lims[1])
    binmids_trunc = binmids[trunc_mask]
    prob_density_trunc = prob_density[trunc_mask]
    dataerr_trunc = dataerr[trunc_mask]

    # Bounds for finding optimal params
    lower_bounds = [0, 0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf]

    # Assigns empty bins an error equal to 1 count to stop divide by zero in curve_fit
    safe_dataerr = np.where(dataerr_trunc == 0, 1.0 / norm_factor, dataerr_trunc)

    fitted_vals, cov = curve_fit(gaussian, binmids_trunc, prob_density_trunc, sigma=safe_dataerr, absolute_sigma=True, bounds=(lower_bounds, upper_bounds), p0=init_guess)

    amp = fitted_vals[0]
    T = fitted_vals[1]
    shift = fitted_vals[2]
    errs = np.sqrt(np.diag(cov))

    if plot:
        fig, ax = plt.subplots(dpi=100)

        if xscale <= 1:
            x_min = trunc_lims[0]
            x_max = trunc_lims[1]
        else:
            # Determine distance from peak to edge of FWHM
            left_dist = abs(shift - binmids_trunc[0])
            right_dist = abs(binmids_trunc[-1] - shift)
            max_dist = max(left_dist, right_dist)
            # Scale domain while keeping peak central
            x_min = shift - (max_dist * xscale)
            x_max = shift + (max_dist * xscale)

        fit_x = np.linspace(x_min, x_max, fit_npoints)
        fit_y = gaussian(fit_x, amp, T, shift)

        if joindata:
            data = ax.errorbar(binmids_trunc, prob_density_trunc, yerr=dataerr_trunc, c='rebeccapurple', marker='.', ls='-', capsize=3, label=r'$p$ dist.', zorder=1000)
            ax.errorbar(binmids, prob_density, yerr=dataerr, alpha = 0.3, c='rebeccapurple', marker='.', ls='-', capsize=3, label=r'$p$ dist.', zorder=999)
        else:
            data = ax.errorbar(binmids_trunc, prob_density_trunc, yerr=dataerr_trunc, c='rebeccapurple', marker='.', ls='none', capsize=3, label=r'$p$ dist.', zorder=1000)
            ax.errorbar(binmids, prob_density, yerr=dataerr, alpha = 0.3, c='rebeccapurple', marker='.', ls='none', capsize=3, label=r'$p$ dist.', zorder=999)
        
        fit, = ax.plot(fit_x, fit_y, c='orange', label='Gaussian fit')

        # Monte carlo sample 1000 curves within 1sigma of optimal params
        sample_params = np.random.multivariate_normal([amp, T, shift], cov, 1000)
        sample_fits = np.array([gaussian(fit_x, *params) for params in sample_params])
        
        # 1-sigma boundaries (15.9th and 84.1st percentiles)
        fiterr_low = np.percentile(sample_fits, 15.9, axis=0)
        fiterr_high = np.percentile(sample_fits, 84.1, axis=0)
        
        # Shade the error region 
        fill = ax.fill_between(fit_x, fiterr_low, fiterr_high, color='orange', alpha=0.3, label=r'$1\sigma$ band')

        ax.grid()
        ax.set_xlim(x_min,x_max)
        ax.set_xlabel(r'$p$ $(\hbar k_{eff})$')
        ax.set_ylabel(r'Probability density')
        ax.legend(handles=[data,fit,fill],labels=[r'$p$ dist.', 'Gaussian fit', r'$1\sigma$ band'])
        plt.show()
    
    if print_vals:
        print(f'Efficiency: {amp*100:.5g} +/- {errs[0]:.5g} %')
        print(f'Temperature: {T*1e6:.5g} +/- {errs[1]*1e6:.5g} muK')
        print(f'Center: {shift:.5g} +/- {errs[2]:.5g} hbark')

    return amp, T, shift, errs