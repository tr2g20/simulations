import numpy as np
from scipy.constants import hbar
from sim_library.constants import k_eff, kb, m, dR
from sim_library.sequences import PulseSequence
from sim_library.hams import gen_ham_free, gen_ham_plus, gen_ham_minus, time_evolve

def simulate_pulses_p_dist(pulse_seq: PulseSequence, Temp: float, no_atoms: int, basis: np.ndarray, initial_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulates a sequence of pulses on an initial state for an ensemble of atoms defined by the given temperature.
    This functions returns an array containing all the possible momentum values as well as an array containing their corresponding weights
    at every time step.

    Args:
        pulse_seq (PulseSequence): The ordered sequence of pulses to be applied to each atom.
        Temp (float): The temperature of the atomic ensemble in Kelvin. Used to determine the velocity distribution width.
        no_atoms (int): The number of atoms in the ensemble.
        basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (numpy.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
                                       

    Returns:
        tuple: A tuple containing four elements:
            - mom_dist (numpy.ndarray): 1D array of all possible momentum values (of size N x basis length).
            - state_fractions (numpy.ndarray): 2D array of the weights (probabilities) of the mom_dist values at every time step (of shape (time steps, N x basis length)).
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

    data = np.zeros((n_steps, len(basis), len(atom_veloc)))

    state_vec = initial_state
    square = np.abs(state_vec)**2
    data[0,:,:] = square[:, np.newaxis] # populate first time step with initial state for every atom

    for v in range(len(atom_veloc)):
        state_vec = initial_state # reset back to initial state for each atom

        pulse_seq.gen_hams(basis= basis, doppler_shift= k_eff*atom_veloc[v])
        hams = pulse_seq.hams
        times = pulse_seq.times
        
        for t in range(1, n_steps):
            state_vec = time_evolve(times[t]-times[t-1], hams[t-1]) @ state_vec # hams is one shorter than times
            square = np.abs(state_vec)**2
            data[t,:,v] = square
        
        
    # remember this is still relative to the initial state    
    init_mom_dist = atom_veloc*m 
    init_mom_dist_tiled = np.tile(init_mom_dist, (len(basis),1))

    basis_tiled = np.transpose(np.tile(basis, (len(atom_veloc),1)))

    # this is a list of all the possible momentum states, 
    # this is fixed and just depends on the initial distribution and the basis
    mom_dist_tiled = init_mom_dist_tiled + (hbar*k_eff*basis_tiled)
    mom_dist = np.ravel(mom_dist_tiled) 

    # this all of the probabilities or weights at each time step
    # the rows line up with mom_dist, then the the column dimension is the time
    state_fractions = data.reshape(data.shape[0],-1)
    
    return mom_dist, state_fractions, rng_state

def simulate_pulses_single_atom(pulse_seq: PulseSequence, basis: np.ndarray, initial_state: np.ndarray, d_shift: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a given sequence of pulses on the given initial state for a single atom.
    By default the atom has no doppler shift but this can be changed with d_shift.
    The output is the mod squared of the final state as a function of time.

    Args:
        pulse_seq (PulseSequence): The ordered sequence of pulses to be applied to each atom.
        basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (numpy.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
                                       
    Returns:
        wave_func (numpy.ndarray): A 2D array of the mod squared wave function at every time step.
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

    wave_func = np.zeros((len(basis), n_steps))

    square = np.abs(state_vec)**2
    wave_func[:,0] = square

    for i in range(1,n_steps):
        state_vec = time_evolve(times[i]-times[i-1], hams[i-1]) @ state_vec # hams is one shorter than times
        square = np.abs(state_vec)**2
        wave_func[:,i] = square

    return np.transpose(wave_func), times


# def simulate_pulses_p_dist(pulse_seq: PulseSequence, Temp: float, no_atoms: int, basis: np.ndarray, initial_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
#     """
#     Simulates a given sequence of pulses on the given initial state for an ensemble of atoms defined by the given temperature.
#     The final momentum state of each atom is calculated and converted to an absolute momentum value to calculate a final momentum distribution.
#     This function only calculates the final state and not intermediate time steps.

#     Args:
#         pulse_seq (PulseSequence): The ordered sequence of pulses to be applied to each atom.
#         Temp (float): The temperature of the atomic ensemble in Kelvin. Used to determine the velocity distribution width.
#         no_atoms (int): The number of atoms in the ensemble.
#         basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
#         initial_state (numpy.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
                                       

#     Returns:
#         tuple: A tuple containing four elements:
#             - final_mom_dist (numpy.ndarray): 1D array of the final absolute momentum values (of size N x basis length).
#             - state_fractions (numpy.ndarray): 1D array of the probabilities corresponding to the final momentum distribution (of size N x basis length).
#             - init_mom_dist (numpy.ndarray): 1D array of the initial momentum distribution (of size N). This distribution is centered on zero and will have to be shifted
#               to represent the absolute momentum of the initial atom ensemble.
#             - rng_state (dict): The state of the numpy random number generator used to generate the initial velocity distribution (containing the seed).

#     Raises:
#         TypeError: If initial_state is not a complex np.ndarray.
#         ValueError: If list of pulses is empty.
#     """
#     # Checks the initial state vector is a complex np.ndarray
#     if not np.issubdtype(initial_state.dtype, np.complexfloating):
#         raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")

#     if len(pulse_seq.pulses) <= 0:
#         raise ValueError("List of pulses is empty")

#     # Defines a new random number generator and saves the state (to be returned), which contains the seed
#     rng = np.random.default_rng()
#     rng_state = dict(rng.bit_generator.state)
    
#     # This distribution is centered on zero for any initial state as the doppler shift is relative to the momentum state
#     sigma = np.sqrt(kb*Temp/m)
#     atom_veloc = rng.normal(loc = 0, scale = sigma, size = no_atoms)

#     data = np.zeros((np.size(basis), np.size(atom_veloc)))

#     for v in range(len(atom_veloc)):
#         state_vec = initial_state

#         for pulse in pulse_seq.pulses:
            
#             H0 = gen_ham_free(basis = basis,
#                   delta_L = pulse.laser_det, 
#                   delta_D = k_eff*atom_veloc[v],
#                   delta_R = dR,
#             )

#             match pulse.type_int:
#                 case 0:
#                     Hint = gen_ham_plus(basis = basis,
#                         phi_L = pulse.phase,
#                         omega_R_plus = pulse.rabi_freq,
#                     )
#                     state_vec = time_evolve(pulse.duration, Hint + H0) @ state_vec
#                 case 1:
#                     Hint = gen_ham_minus(basis = basis,
#                         phi_L = pulse.phase,
#                         omega_R_minus = pulse.rabi_freq,
#                     )
#                     state_vec = time_evolve(pulse.duration, Hint + H0) @ state_vec
#                 case 2:
#                     state_vec = time_evolve(pulse.duration, H0) @ state_vec
        
#         square = np.abs(state_vec)**2
#         data[:,v] = square


#     init_mom_dist = atom_veloc*m
#     init_mom_dist_tiled = np.tile(init_mom_dist, (len(basis),1))

#     basis_tiled = np.transpose(np.tile(basis, (len(atom_veloc),1)))

#     final_mom_dist_tiled = init_mom_dist_tiled + (hbar*k_eff*basis_tiled)

#     final_mom_dist = np.ravel(final_mom_dist_tiled)

#     state_fractions = np.ravel(data)
    
#     return final_mom_dist, state_fractions, init_mom_dist, rng_state

