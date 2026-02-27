import numpy as np
from scipy.constants import hbar
from sim_library.constants import k_eff, kb, m, dR
from sim_library.sequences import PulseSequence, RR3_gate
from sim_library.hams import gen_ham_free, gen_ham_plus, gen_ham_minus, time_evolve
from sim_library.data_io import save_p_dists

def simulate_pulses_p_dist(pulse_seq: PulseSequence, Temp: float, no_atoms: int, basis: np.ndarray, initial_state: np.ndarray):
    """
    Simulates a sequence of pulses on an initial state for an ensemble of atoms defined by the given temperature.
    This functions returns the momentum distribution along with the weights (probabilities) corresponding to each momentum state.

    Args:
        pulse_seq (PulseSequence): The ordered sequence of pulses to be applied to each atom.
        Temp (float): The temperature of the atomic ensemble in Kelvin. Used to determine the velocity distribution width.
        no_atoms (int): The number of atoms in the ensemble.
        basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (numpy.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
                                       

    Returns:
        tuple: A tuple containing five elements:
            - mom_dist (numpy.ndarray): 1D array of all possible momentum values (of size N*basis length). Use this to plot a histogram.
            - state_fractions (numpy.ndarray): 2D array of the weights (probabilities) of the mom_dist values at every time step (of shape (time steps, N*basis length)). Use this to plot a histogram.
            - mom_dist_grid (numpy.ndarray): 2D array of all possible momentum values (of shape (basis length, N)).
            - state_fractions_grid (numpy.ndarray): 3D array of the weights (probabilities) of the mom_dist_grid values at every time step (of shape (time steps, basis length, N)).
            - rng_state (dict): The state of the numpy random number generator used to generate the initial velocity distribution (containing the seed).

    Raises:
        TypeError: If initial_state is not a complex np.ndarray.
        ValueError: If list of pulses is empty.
    """
    # Checks the initial state vector is a complex np.ndarray
    if not np.issubdtype(initial_state.dtype, np.complexfloating):
        raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")

    if len(pulse_seq.pulses) <= 0:
        raise ValueError("List of pulses is empty")

    # Defines a new random number generator and saves the state (to be returned), which contains the seed
    rng = np.random.default_rng()
    rng_state = dict(rng.bit_generator.state)
    
    # This distribution is centered on zero for any initial state as the doppler shift is relative to the momentum state
    sigma = np.sqrt(kb*Temp/m)
    atom_veloc = rng.normal(loc = 0, scale = sigma, size = no_atoms)

    n_steps = pulse_seq.get_n_steps()

    state_fractions_grid = np.zeros((n_steps, len(basis), len(atom_veloc)))

    state_vec = initial_state
    square = np.abs(state_vec)**2
    state_fractions_grid[0,:,:] = square[:, np.newaxis] # populate first time step with initial state for every atom

    for v in range(len(atom_veloc)):
        state_vec = initial_state # reset back to initial state for each atom

        pulse_seq.gen_hams(basis= basis, doppler_shift= k_eff*atom_veloc[v])
        hams = pulse_seq.hams
        times = pulse_seq.times
        
        for t in range(1, n_steps):
            state_vec = time_evolve(times[t]-times[t-1], hams[t-1]) @ state_vec # hams is one shorter than times
            square = np.abs(state_vec)**2
            state_fractions_grid[t,:,v] = square
        
        
    # remember this is still relative to the initial state    
    init_mom_dist = atom_veloc*m 
    init_mom_dist_tiled = np.tile(init_mom_dist, (len(basis),1))

    basis_tiled = np.transpose(np.tile(basis, (len(atom_veloc),1)))

    # this is a list of all the possible momentum states, 
    # this is fixed and just depends on the initial distribution and the basis
    mom_dist_grid = init_mom_dist_tiled + (hbar*k_eff*basis_tiled)
    mom_dist = np.ravel(mom_dist_grid) 

    # this all of the probabilities or weights at each time step
    # the rows line up with mom_dist, then the the column dimension is the time
    state_fractions = state_fractions_grid.reshape(state_fractions_grid.shape[0],-1)
    
    return mom_dist, state_fractions, mom_dist_grid, state_fractions_grid, rng_state

def simulate_pulses_p_dist_custom(pulse_seq: PulseSequence, init_mom_dist: np.ndarray, basis: np.ndarray, initial_state: np.ndarray):
    """
    Simulates a sequence of pulses on an initial state for an ensemble of atoms with a given momentum distribution.
    This functions returns the momentum distribution along with the weights (probabilities) corresponding to each momentum state.

    Args:
        pulse_seq (PulseSequence): The ordered sequence of pulses to be applied to each atom.
        init_mom_dist (numpy.ndarray): 1D array of initial momentum distribution (of size N).
        basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (numpy.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
                                       

    Returns:
        tuple: A tuple containing four elements:
            - mom_dist (numpy.ndarray): 1D array of all possible momentum values (of size N*basis length). Use this to plot a histogram.
            - state_fractions (numpy.ndarray): 2D array of the weights (probabilities) of the mom_dist values at every time step (of shape (time steps, N*basis length)). Use this to plot a histogram.
            - mom_dist_grid (numpy.ndarray): 2D array of all possible momentum values (of shape (basis length, N)).
            - state_fractions_grid (numpy.ndarray): 3D array of the weights (probabilities) of the mom_dist_grid values at every time step (of shape (time steps, basis length, N)).

    Raises:
        TypeError: If initial_state is not a complex np.ndarray.
        ValueError: If list of pulses is empty.
    """
    # Checks the initial state vector is a complex np.ndarray
    if not np.issubdtype(initial_state.dtype, np.complexfloating):
        raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")

    if len(pulse_seq.pulses) <= 0:
        raise ValueError("List of pulses is empty")

    atom_veloc = init_mom_dist/m

    n_steps = pulse_seq.get_n_steps()

    state_fractions_grid = np.zeros((n_steps, len(basis), len(atom_veloc)))

    state_vec = initial_state
    square = np.abs(state_vec)**2
    state_fractions_grid[0,:,:] = square[:, np.newaxis] # populate first time step with initial state for every atom

    for v in range(len(atom_veloc)):
        state_vec = initial_state # reset back to initial state for each atom

        pulse_seq.gen_hams(basis= basis, doppler_shift= k_eff*atom_veloc[v])
        hams = pulse_seq.hams
        times = pulse_seq.times
        
        for t in range(1, n_steps):
            state_vec = time_evolve(times[t]-times[t-1], hams[t-1]) @ state_vec # hams is one shorter than times
            square = np.abs(state_vec)**2
            state_fractions_grid[t,:,v] = square
        
        
    # remember this is still relative to the initial state    
    init_mom_dist_tiled = np.tile(init_mom_dist, (len(basis),1))

    basis_tiled = np.transpose(np.tile(basis, (len(atom_veloc),1)))

    # this is a list of all the possible momentum states, 
    # this is fixed and just depends on the initial distribution and the basis
    mom_dist_grid = init_mom_dist_tiled + (hbar*k_eff*basis_tiled)
    mom_dist = np.ravel(mom_dist_grid) 

    # this all of the probabilities or weights at each time step
    # the rows line up with mom_dist, then the the column dimension is the time
    state_fractions = state_fractions_grid.reshape(state_fractions_grid.shape[0],-1)
    
    return mom_dist, state_fractions, mom_dist_grid, state_fractions_grid

def simulate_pulses_single_atom(pulse_seq: PulseSequence, basis: np.ndarray, initial_state: np.ndarray, d_shift: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a given sequence of pulses on the given initial state for a single atom.
    By default the atom has no doppler shift but this can be changed with d_shift.
    The output is the state vector as a function of time.

    Args:
        pulse_seq (PulseSequence): The ordered sequence of pulses to be applied to each atom.
        basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (numpy.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
                                       
    Returns:
        wave_func (numpy.ndarray): A 2D array of the wave function at every time step.
        times (nump.ndarray): A 1D array of each time step (in seconds).

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

    for i in range(1,n_steps):
        state_vec = time_evolve(times[i]-times[i-1], hams[i-1]) @ state_vec # hams is one shorter than times
        wave_func[:,i] = state_vec

    return np.transpose(wave_func), times

def p_shift_random_walk(ground_state_ratio: float):
    """Performs a random walk to simulate how many photon absorptions and spontaneous emissions before an atom is pumped to a dark state.

    Args:
        ground_state_ratio (float): The probability of the atom decaying to the ground (dark) state after being excited. Calculated from transition strengths.

    Returns:
        tuple: A tuple containing:
            - p (float): The total accumulated momentum shift.
            - n (int): The total number of photon scatters that occur.
    """
    p=0
    n=0
    walking = True
    while walking:
        p += np.random.choice([-1,1])
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

def simulate_alg_cooling(basis: np.ndarray, time_steps: int, n_atoms: int, temp: float, initial_state: np.ndarray, cycles: int, rabi_freq: float, save_to_file = False):
    
    if not np.issubdtype(initial_state.dtype, np.complexfloating):
        raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")
    
    rng = np.random.default_rng()
    rng_state = dict(rng.bit_generator.state)
    
    sigma = np.sqrt(kb*temp/m)
    p_dist = m*rng.normal(loc = 0, scale = sigma, size = n_atoms)

    RR3 = RR3_gate(rabi_freq=rabi_freq, time_steps=time_steps)

    moms_list = [p_dist]
    fracs_list = [np.ones(len(p_dist))]

    moms, fracs, mom_grid, frac_grid = simulate_pulses_p_dist_custom(pulse_seq=RR3, init_mom_dist=p_dist, basis=basis, initial_state=initial_state)

    moms_list.append(moms)
    fracs_list.append(fracs[-1])

    moms = simulate_optical_pumping(basis=basis, init_mom_dist_grid=mom_grid, state_fracs_grid=frac_grid[-1,:,:])

    moms_list.append(moms)
    fracs_list.append(np.ones(len(moms)))

    for i in range(cycles-1):
        moms, fracs, mom_grid, frac_grid = simulate_pulses_p_dist_custom(pulse_seq=RR3, init_mom_dist=moms, basis=basis, initial_state=initial_state)

        moms_list.append(moms)
        fracs_list.append(fracs[-1])

        moms = simulate_optical_pumping(basis=basis, init_mom_dist_grid=mom_grid, state_fracs_grid=frac_grid[-1,:,:])

        moms_list.append(moms)
        fracs_list.append(np.ones(len(moms)))
    
    if save_to_file:
        save_p_dists() # continue from here

    return moms_list, fracs_list, rng_state