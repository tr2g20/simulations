import numpy as np
from scipy.constants import hbar
from sim_library.constants import k_eff, kb, m, dR, pi
from sim_library.sequences import PulseSequence, RR3_gate
from sim_library.hams import time_evolve, gen_ham_free, gen_ham_minus, gen_ham_plus
from sim_library.data_io import save_p_dists
from datetime import date
from pathlib import Path
from numba import njit

@njit
def gen_state_fractions(state_fractions_grid: np.ndarray, state_vec: np.ndarray, 
                        basis: np.ndarray, initial_state: np.ndarray, init_mom_dist: np.ndarray, beam_profile: np.ndarray, rabis: np.ndarray, detunings: np.ndarray, phases: np.ndarray, times: np.ndarray, pulse_types: np.ndarray):
    """
    Core Numba-compiled engine that calculates the state fractions for an ensemble of atoms over a sequence of pulses.
    This function modifies the pre-allocated workspaces in-place to ensure zero-allocation performance during high-frequency loops.

    Args:
        state_fractions_grid (np.ndarray): 3D pre-allocated workspace of shape (time steps, basis length, N) where probability weights will be written. Modified in-place.
        state_vec (np.ndarray): 1D pre-allocated buffer array of shape (basis length,) used for time evolution calculations. Modified in-place.
        basis (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (np.ndarray): The initial momentum state vector of a single atom (elements are np.complex128).
        init_mom_dist (np.ndarray): 1D array of the initial momentum distribution of the ensemble.
        beam_profile (np.ndarray): 1D array representing the beam profile intensity modifier for each atom.
        rabis (np.ndarray): 1D array of Rabi frequencies at each discrete time step in 2pi*Hz.
        detunings (np.ndarray): 1D array of laser detunings at each discrete time step in 2pi*Hz.
        phases (np.ndarray): 1D array of laser phases at each discrete time step.
        times (np.ndarray): 1D array of absolute time points for the sequence.
        pulse_types (np.ndarray): 1D array of pulse type identifiers (0 for Up, 1 for Down) at each time step.
    """
    
    n_steps = len(phases) + 1
    n_atoms = len(init_mom_dist)
    
    state_fractions_grid[0, :, :] = np.abs(initial_state)[:, np.newaxis]**2

    for v in range(n_atoms):
        state_vec[:] = initial_state # reset back to initial state for each atom

        for t in range(1, n_steps):
            atom_veloc = init_mom_dist[v]/m
            ham = gen_ham_free(basis, detunings[t-1], k_eff*atom_veloc, dR)

            adjusted_rabi = beam_profile[v]*rabis[t-1] # increases or decreases the preset rabi frequency to simulate intensity noise
            
            if pulse_types[t-1] == 0:    # UpPulse
                ham += gen_ham_plus(basis, phases[t-1], adjusted_rabi)
            elif pulse_types[t-1] == 1:  # DownPulse
                ham += gen_ham_minus(basis, phases[t-1], adjusted_rabi)

            state_vec = time_evolve(state_vec, times[t]-times[t-1], ham) # times is one longer than other arrays
            state_fractions_grid[t,:,v] = np.abs(state_vec)**2

def simulate_pulses_p_dist(pulse_seq: PulseSequence, Temp: float, no_atoms: int, basis: np.ndarray, initial_state: np.ndarray, beam_profile: np.ndarray):
    """
    Simulation of a sequence of pulses on an atomic ensemble defined by a given temperature.

    Args:
        pulse_seq (PulseSequence): Represents an ordered sequence of Pulse objects.
        Temp (float): The temperature of the atomic ensemble in Kelvin. Used to determine the velocity distribution width.
        no_atoms (int): The number of atoms in the ensemble.
        basis (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (np.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
        beam_profile (np.ndarray): The 1D array representing spatial intensity noise/profile for each atom.

    Returns:
        tuple: A tuple containing five elements:
            - mom_dist (np.ndarray): 1D array of all possible momentum values.
            - state_fractions (np.ndarray): 2D array of the corresponding weights (probabilities) of mom_dist at every time step.
            - mom_dist_grid (np.ndarray): 2D array of all possible momentum values (shape: basis length, N).
            - state_fractions_grid (np.ndarray): 3D array of the weights corresponding to mom_dist_grid (shape: time steps, basis length, N).
            - rng_state (dict): The state of the numpy random number generator used for the initial velocity distribution.

    Raises:
        TypeError: If initial_state is not a complex np.ndarray.
        ValueError: If the list of pulses is empty.
        RuntimeError: If pulse sequence arrays have not been built.
    """
    # Checks the initial state vector is a complex np.ndarray
    if not np.issubdtype(initial_state.dtype, np.complexfloating):
        raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")

    if len(pulse_seq.pulses) <= 0:
        raise ValueError("List of pulses is empty")
    
    if len(pulse_seq.phases) == 0:
        raise RuntimeError("Pulse sequence parameters are empty. Call sequence.build_seq() to initialise arrays after adding pulses.")

    # Defines a new random number generator and saves the state (to be returned), which contains the seed
    rng = np.random.default_rng()
    rng_state = dict(rng.bit_generator.state)
    
    sigma = np.sqrt(kb*Temp/m)
    init_mom_dist = m*rng.normal(loc = 0, scale = sigma, size = no_atoms)

    n_steps = len(pulse_seq.phases) + 1

    ### Globally allocated arrays ###
    state_fractions_grid = np.empty((n_steps, len(basis), no_atoms), dtype=np.float64)
    state_vec_buffer = np.empty(len(basis), dtype=np.complex128)
    ###

    gen_state_fractions(state_fractions_grid=state_fractions_grid, state_vec=state_vec_buffer,
                        basis=basis, initial_state=initial_state, init_mom_dist=init_mom_dist, beam_profile=beam_profile, 
                        rabis=pulse_seq.rabis, detunings=pulse_seq.detunings, phases=pulse_seq.phases, times=pulse_seq.times, pulse_types=pulse_seq.pulse_types)
        
    mom_dist_grid = init_mom_dist[np.newaxis, :] + (hbar * k_eff * basis[:, np.newaxis])
    
    mom_dist = np.ravel(mom_dist_grid) # do i need to copy these?
    # state_fractions = np.ravel(state_fractions_grid[-1, :, :])
    state_fractions = state_fractions_grid.reshape(n_steps, -1)
    
    return mom_dist, state_fractions, mom_dist_grid, state_fractions_grid, rng_state

def simulate_pulses_p_dist_custom(pulse_seq: PulseSequence, init_mom_dist: np.ndarray, no_atoms: int, basis: np.ndarray, initial_state: np.ndarray, beam_profile: np.ndarray):
    """
    Simulation of a sequence of pulses on an atomic ensemble. Momentum distribution is passed in as variable.

    Args:
        pulse_seq (PulseSequence): Represents an ordered sequence of Pulse objects.
        init_mom_dist (np.ndarray): 1D array of the initial momentum distribution of the ensemble.
        no_atoms (int): The number of atoms in the ensemble.
        basis (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (np.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
        beam_profile (np.ndarray): The 1D array representing spatial intensity noise/profile for each atom.

    Returns:
        tuple: A tuple containing four elements:
            - mom_dist (np.ndarray): 1D array of all possible momentum values.
            - state_fractions (np.ndarray): 2D array of the corresponding weights (probabilities) of mom_dist at every time step.
            - mom_dist_grid (np.ndarray): 2D array of all possible momentum values (shape: basis length, N).
            - state_fractions_grid (np.ndarray): 3D array of the weights corresponding to mom_dist_grid (shape: time steps, basis length, N).

    Raises:
        TypeError: If initial_state is not a complex np.ndarray.
        ValueError: If the list of pulses is empty.
        RuntimeError: If pulse sequence arrays have not been built.
    """

    # Checks the initial state vector is a complex np.ndarray
    if not np.issubdtype(initial_state.dtype, np.complexfloating):
        raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")

    if len(pulse_seq.pulses) <= 0:
        raise ValueError("List of pulses is empty")
    
    if len(pulse_seq.phases) == 0:
        raise RuntimeError("Pulse sequence parameters are empty. Call sequence.build_seq() to initialise arrays after adding pulses.")

    n_steps = len(pulse_seq.phases) + 1

    ### Globally allocated arrays ###
    state_fractions_grid = np.empty((n_steps, len(basis), no_atoms), dtype=np.float64)
    state_vec_buffer = np.empty(len(basis), dtype=np.complex128)
    ###

    gen_state_fractions(state_fractions_grid=state_fractions_grid, state_vec=state_vec_buffer,
                        basis=basis, initial_state=initial_state, init_mom_dist=init_mom_dist, beam_profile=beam_profile, 
                        rabis=pulse_seq.rabis, detunings=pulse_seq.detunings, phases=pulse_seq.phases, times=pulse_seq.times, pulse_types=pulse_seq.pulse_types)
        
    mom_dist_grid = init_mom_dist[np.newaxis, :] + (hbar * k_eff * basis[:, np.newaxis])
    
    mom_dist = np.ravel(mom_dist_grid) # do i need to copy these?
    state_fractions = np.ravel(state_fractions_grid[-1, :, :])
    
    return mom_dist, state_fractions, mom_dist_grid, state_fractions_grid

def simulate_pulses_single_atom(pulse_seq: PulseSequence, basis: np.ndarray, initial_state: np.ndarray, d_shift: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a given sequence of pulses on the given initial state for a single atom.
    By default the atom has no doppler shift but this can be changed with d_shift.
    The output is the state vector as a function of time.

    Args:
        pulse_seq (PulseSequence): The ordered sequence of pulses to be applied to each atom.
        basis (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (np.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
                                       
    Returns:
        wave_func (np.ndarray): A 2D array of the wave function at every time step.
        times (np.ndarray): A 1D array of each time step (in seconds).

    Raises:
        TypeError: If initial_state is not a complex np.ndarray.
        ValueError: If list of pulses is empty.
    """
    # Checks the initial state vector is a complex np.ndarray
    if not np.issubdtype(initial_state.dtype, np.complexfloating):
        raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")

    if len(pulse_seq.pulses) <= 0:
        raise ValueError("List of pulses is empty")
    
    pulse_seq.gen_hams(basis= basis, doppler_shift= d_shift)
    hams = pulse_seq.hams
    times = pulse_seq.times

    n_steps = len(times)

    state_vec = initial_state

    wave_func = np.zeros((len(basis), n_steps), dtype=np.complex128)

    wave_func[:,0] = state_vec

    for t in range(1,n_steps):
        state_vec = time_evolve(state_vec= state_vec, dt = times[t]-times[t-1], H = hams[t-1]) # hams is one shorter than times
        wave_func[:,t] = state_vec

    return np.transpose(wave_func), times

@njit
def p_shift_random_walk(ground_state_ratio: float):
    """Performs a random walk to simulate how many photon absorptions and spontaneous emissions before an atom is pumped to a dark state.

    Args:
        ground_state_ratio (float): The probability of the atom decaying to the ground (dark) state after being excited. Calculated from transition strengths.

    Returns:
        tuple: A tuple containing:
            - p (float): The total accumulated momentum shift.
            - n (int): The total number of photon scatters that occur.
    """
    p=0.0
    n=0
    walking = True

    while walking:
        p += 1.0 if np.random.random() > 0.5 else -1.0
        p += np.random.uniform(-1,1)
        n += 1
        a = np.random.random()
        if a < ground_state_ratio:
            walking = False
    return p, n
 
def simulate_optical_pumping(basis: np.ndarray, init_mom_dist_grid: np.ndarray, state_fracs_grid: np.ndarray, pumping_route: str = 'f3'):
    """Simulates the effect of optical pumping on the momentum distribution of an atomic ensemble.

    Args:
        basis (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        init_mom_dist_grid (np.ndarray): 2D array of the initial momentum distribution 
            grid (of shape (basis length, N)).
        state_fracs_grid (np.ndarray): 2D array of state probabilities corresponding 
            to the basis states for each atom (of shape (basis length, N)).
        pumping_route (str, optional): Upper hyperfine state that the pumping route takes. Options are 'f3' (default) and 'f2', corresponding to F'=3 and F'=2.

    Returns:
        np.ndarray: 1D array of the new momentum distribution for the atom ensemble (of length N).
    """
    n_atoms = init_mom_dist_grid.shape[1]
    new_mom_dist = np.zeros(n_atoms)
    pumping_momentum = hbar*k_eff*0.5 #assume transition energy is half of raman transition

    if pumping_route == 'f3':
        ground_state_ratio = 0.528
    else:
        ground_state_ratio = 0.831

    for i in range(n_atoms):
        basis_index = np.random.choice(a=np.arange(len(basis)), p=state_fracs_grid[:,i])
        if basis[basis_index] % 2 == 0:
            new_mom_dist[i] += init_mom_dist_grid[basis_index, i]
        else:
            p_shift, _ = p_shift_random_walk(ground_state_ratio=ground_state_ratio)
            p_shift = p_shift*pumping_momentum
            new_mom_dist[i] += init_mom_dist_grid[basis_index, i] + p_shift
            
    return new_mom_dist

def simulate_alg_cooling(basis: np.ndarray, sequence: PulseSequence, n_atoms: int, temp: float, initial_state: np.ndarray, beam_profile: np.ndarray, cycles: int, rabi_freq: float, pumping_route: str, alternating: bool = False, save_dir: str = ''):
    """
    Simulates cooling cycles of a given pulse sequence on an ensemble of atoms defined by temp and n_atoms.

    Args:
        basis (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        sequence (PulseSequence): Represents an ordered sequence of Pulse objects.
        n_atoms (int): Number of atoms in the ensemble.
        temp (float): Initial temperature in Kelvin.
        initial_state (np.ndarray): The initial momentum state vector of a single atom (elements are np.complex128).
        beam_profile (np.ndarray): 1D array of spatial intensity noise across the ensemble.
        cycles (int): Number of algorithmic cooling cycles to perform.
        rabi_freq (float): The base Rabi frequency in 2pi*Hz.
        pumping_route (str): Optical pumping route (e.g., 'f3' or 'f2').
        alternating (bool, optional): If True, multiplies momentum by -1 each cycle to flip the distribution. Defaults to False.
        save_dir (str, optional): Directory path to save the .h5 output file. If empty, data is not saved.

    Returns:
        tuple: A tuple containing:
            - moms_list (list[np.ndarray]): List of 1D arrays recording the complete momentum distribution after each pulse sequence and each optical pumping.
            - fracs_list (list[np.ndarray]): List of 1D arrays recording the probability fractions corresponding to moms_list.
            - rng_state (dict): The seed dictionary from the random number generator.
    """

    if not np.issubdtype(initial_state.dtype, np.complexfloating):
        raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")
    
    if len(sequence.pulses) <= 0:
        raise ValueError("List of pulses is empty")
    
    if len(sequence.phases) == 0:
        raise RuntimeError("Pulse sequence parameters are empty. Call sequence.build_seq() to initialise arrays after adding pulses.")

    rng = np.random.default_rng()
    rng_state = dict(rng.bit_generator.state)
    
    sigma = np.sqrt(kb*temp/m)
    p_dist = m*rng.normal(loc = 0, scale = sigma, size = n_atoms)

    n_steps = len(sequence.phases) + 1 # number of time steps in a sequence

    ### Globally allocated arrays ###
    # This is to optimise memory usage and stop unecessary copies of arrays being made
    # These get passed into each simulation function, by reference so no returns are needed
    state_fractions_grid = np.empty((n_steps, len(basis), n_atoms), dtype=np.float64)
    mom_dist_grid = np.empty((len(basis), n_atoms), dtype=np.float64)
    state_vec_buffer = np.empty(len(basis), dtype=np.complex128)
    ###

    ### convert initial p dist + initial state into a momentum distribution and weights array
    mom_dist_grid[:] = p_dist[np.newaxis, :] + (hbar * k_eff * basis[:, np.newaxis])
    square = np.abs(initial_state)**2
    init_state_fractions = (square[:, np.newaxis] * np.ones((1, n_atoms))).ravel()
    ###

    moms_list = [mom_dist_grid.ravel().copy()]
    fracs_list = [init_state_fractions]

    gen_state_fractions(state_fractions_grid=state_fractions_grid, state_vec=state_vec_buffer, 
                        basis=basis, initial_state=initial_state, init_mom_dist=p_dist, beam_profile=beam_profile, rabis=sequence.rabis, detunings=sequence.detunings, phases=sequence.phases, times=sequence.times, pulse_types=sequence.pulse_types)          
    # this is a list of all the possible momentum states, 
    # this is fixed and just depends on the initial distribution and the basis
    mom_dist_grid[:] = p_dist[np.newaxis, :] + (hbar * k_eff * basis[:, np.newaxis])

    moms_list.append(mom_dist_grid.ravel().copy())
    fracs_list.append(state_fractions_grid[-1, :, :].ravel().copy())

    current_moms = simulate_optical_pumping(basis=basis, init_mom_dist_grid=mom_dist_grid, state_fracs_grid=state_fractions_grid[-1,:,:], pumping_route=pumping_route)

    moms_list.append(current_moms)
    fracs_list.append(np.ones(len(current_moms)))

    zero_index = np.argmax(basis == 0) # returns index of 0
    reset_state = np.zeros(shape=len(basis), dtype=np.complex128) # we reset the atoms all into the ground state (p=0) and their momentum state info
    reset_state[zero_index] = 1                                   # is then purely contained in the doppler shift from the input momentum distribution for the next cycle

    multiplier = 1

    for i in range(cycles-1):

        if alternating:
            current_moms = current_moms*-1
            multiplier = (-1)**(i+1) # corrects the flip in the moms_list

        gen_state_fractions(state_fractions_grid=state_fractions_grid, state_vec=state_vec_buffer, 
                        basis=basis, initial_state=reset_state, init_mom_dist=current_moms, beam_profile=beam_profile, rabis=sequence.rabis, detunings=sequence.detunings, phases=sequence.phases, times=sequence.times, pulse_types=sequence.pulse_types)   
        mom_dist_grid[:] = current_moms[np.newaxis, :] + (hbar * k_eff * basis[:, np.newaxis])

        moms_list.append(mom_dist_grid.ravel().copy()*multiplier)
        fracs_list.append(state_fractions_grid[-1, :, :].ravel().copy())

        current_moms = simulate_optical_pumping(basis=basis, 
                                                init_mom_dist_grid=mom_dist_grid, state_fracs_grid=state_fractions_grid[-1,:,:], 
                                                pumping_route=pumping_route)

        moms_list.append(current_moms*multiplier)
        fracs_list.append(np.ones(len(current_moms)))
    
    if save_dir != '':
        date_str = date.today().isoformat()
        file_name = f"{date_str}_algcooling_{pumping_route}_{cycles}cycles_{temp*1e6:.0f}muK_{n_atoms}atoms_{rabi_freq/(2*pi):.0g}Hz.h5"
        file_path = Path(save_dir) / file_name
        save_p_dists(file_path=file_path, p_list=moms_list, weights_list=fracs_list, init_temp=temp, n_atoms=n_atoms, basis=basis, init_state=initial_state, cycles=cycles, pumping_route=pumping_route, date=date_str)

    return moms_list, fracs_list, rng_state

def simulate_alg_cooling_custom(basis: np.ndarray, sequence: PulseSequence, p_dist: np.ndarray, initial_state: np.ndarray, beam_profile: np.ndarray, cycles: int, rabi_freq: float, pumping_route: str, alternating: bool = False, save_dir: str = ''):
    """
    Simulates cooling cycles of a given pulse sequence on an ensemble of atoms defined by a given momentum distribution.

    Args:
        basis (np.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        sequence (PulseSequence): Represents an ordered sequence of Pulse objects.
        p_dist (np.ndarray): 1D array of the custom initial momentum distribution.
        initial_state (np.ndarray): The initial momentum state vector of a single atom (elements are np.complex128).
        beam_profile (np.ndarray): 1D array of spatial intensity noise across the ensemble.
        cycles (int): Number of algorithmic cooling cycles to perform.
        rabi_freq (float): The base Rabi frequency in 2pi*Hz.
        pumping_route (str): Optical pumping route (e.g., 'f3' or 'f2').
        alternating (bool, optional): If True, multiplies momentum by -1 each cycle to flip the distribution. Defaults to False.
        save_dir (str, optional): Directory path to save the .h5 output file. If empty, data is not saved.

    Returns:
        tuple: A tuple containing:
            - moms_list (list[np.ndarray]): List of 1D arrays recording the complete momentum distribution at key stages.
            - fracs_list (list[np.ndarray]): List of 1D arrays recording the probability fractions corresponding to moms_list.
    """

    if not np.issubdtype(initial_state.dtype, np.complexfloating):
        raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")
    
    if len(sequence.pulses) <= 0:
        raise ValueError("List of pulses is empty")
    
    if len(sequence.phases) == 0:
        raise RuntimeError("Pulse sequence parameters are empty. Call sequence.build_seq() to initialise arrays after adding pulses.")
    
    n_atoms = len(p_dist)
    n_steps = len(sequence.phases) + 1

    ### Globally allocated arrays ###
    state_fractions_grid = np.empty((n_steps, len(basis), n_atoms), dtype=np.float64)
    mom_dist_grid = np.empty((len(basis), n_atoms), dtype=np.float64)
    state_vec_buffer = np.empty(len(basis), dtype=np.complex128)
    ###

    ### convert initial p dist + initial state into a momentum distribution and weights array
    mom_dist_grid[:] = p_dist[np.newaxis, :] + (hbar * k_eff * basis[:, np.newaxis])
    square = np.abs(initial_state)**2
    init_state_fractions = (square[:, np.newaxis] * np.ones((1, n_atoms))).ravel()
    ###

    moms_list = [mom_dist_grid.ravel().copy()]
    fracs_list = [init_state_fractions]

    gen_state_fractions(state_fractions_grid=state_fractions_grid, state_vec=state_vec_buffer, 
                        basis=basis, initial_state=initial_state, init_mom_dist=p_dist, beam_profile=beam_profile, rabis=sequence.rabis, detunings=sequence.detunings, phases=sequence.phases, times=sequence.times, pulse_types=sequence.pulse_types)          
    # this is a list of all the possible momentum states, 
    # this is fixed and just depends on the initial distribution and the basis
    mom_dist_grid[:] = p_dist[np.newaxis, :] + (hbar * k_eff * basis[:, np.newaxis])

    moms_list.append(mom_dist_grid.ravel().copy())
    fracs_list.append(state_fractions_grid[-1, :, :].ravel().copy())

    current_moms = simulate_optical_pumping(basis=basis, init_mom_dist_grid=mom_dist_grid, state_fracs_grid=state_fractions_grid[-1,:,:], pumping_route=pumping_route)

    moms_list.append(current_moms)
    fracs_list.append(np.ones(len(current_moms)))

    zero_index = np.argmax(basis == 0) 
    reset_state = np.zeros(shape=len(basis), dtype=np.complex128) 
    reset_state[zero_index] = 1                                   

    multiplier = 1

    for i in range(cycles-1):

        if alternating: 
            current_moms = current_moms * -1
            multiplier = (-1)**(i+1) 

        gen_state_fractions(state_fractions_grid=state_fractions_grid, state_vec=state_vec_buffer, 
                        basis=basis, initial_state=reset_state, init_mom_dist=current_moms, beam_profile=beam_profile, rabis=sequence.rabis, detunings=sequence.detunings, phases=sequence.phases, times=sequence.times, pulse_types=sequence.pulse_types)   
        mom_dist_grid[:] = current_moms[np.newaxis, :] + (hbar * k_eff * basis[:, np.newaxis])

        moms_list.append(mom_dist_grid.ravel().copy() * multiplier)
        fracs_list.append(state_fractions_grid[-1, :, :].ravel().copy())

        current_moms = simulate_optical_pumping(basis=basis, 
                                                init_mom_dist_grid=mom_dist_grid, state_fracs_grid=state_fractions_grid[-1,:,:], 
                                                pumping_route=pumping_route)

        moms_list.append(current_moms * multiplier)
        fracs_list.append(np.ones(len(current_moms)))
    
    if save_dir != '':
        date_str = date.today().isoformat()
        file_name = f"{date_str}_algcooling_{pumping_route}_{cycles}cycles_{n_atoms}atoms_{rabi_freq/(2*pi):.0g}Hz.h5"
        file_path = Path(save_dir) / file_name
        save_p_dists(file_path=file_path, p_list=moms_list, weights_list=fracs_list, init_temp='na', n_atoms=n_atoms, basis=basis, init_state=initial_state, cycles=cycles, pumping_route=pumping_route, date=date_str)

    return moms_list, fracs_list
