import numpy as np
from scipy.constants import hbar
from sim_library.constants import k_eff, kb, m, dR
from sim_library.sequences import PulseSequence
from sim_library.hams import gen_ham_free, gen_ham_plus, gen_ham_minus, time_evolve

def simulate_pulses_p_dist(pulse_seq: PulseSequence, Temp: float, no_atoms: int, basis: np.ndarray, initial_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Simulates a given sequence of pulses on the given initial state for an ensemble of atoms defined by the given temperature.
    The final momentum state of each atom is calculated and converted to an absolute momentum value to calculate a final momentum distribution.
    This function only calculates the final state and not intermediate time steps.

    Args:
        pulse_seq (PulseSequence): The ordered sequence of pulses to be applied to each atom.
        Temp (float): The temperature of the atomic ensemble in Kelvin. Used to determine the velocity distribution width.
        no_atoms (int): The number of atoms in the ensemble.
        basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
        initial_state (numpy.ndarray): The initial momentum state vector of a single atom (elements are np.complex128). 
                                       

    Returns:
        tuple: A tuple containing four elements:
            - final_mom_dist (numpy.ndarray): 1D array of the final absolute momentum values (of size N x basis length).
            - state_fractions (numpy.ndarray): 1D array of the probabilities corresponding to the final momentum distribution (of size N x basis length).
            - init_mom_dist (numpy.ndarray): 1D array of the initial momentum distribution (of size N). This distribution is centered on zero and will have to be shifted
              to represent the absolute momentum of the initial atom ensemble.
            - rng_state (dict): The state of the numpy random number generator used to generate the initial velocity distribution (containing the seed).

    Raises:
        TypeError: If initial_state is not a complex np.ndarray.
    """
    # Checks the initial state vector is a complex np.ndarray
    if not np.issubdtype(initial_state.dtype, np.complexfloating):
        raise TypeError(f"The 'initial_state' array must have a complex dtype (e.g., np.complex128), but received {initial_state.dtype}.")

    # Defines a new random number generator and saves the state (to be returned), which contains the seed
    rng = np.random.default_rng()
    rng_state = dict(rng.bit_generator.state)
    
    # This distribution is centered on zero for any initial state as the doppler shift is relative to the momentum state
    sigma = np.sqrt(kb*Temp/m)
    atom_veloc = rng.normal(loc = 0, scale = sigma, size = no_atoms)

    data = np.zeros((np.size(basis), np.size(atom_veloc)))

    for v in range(len(atom_veloc)):
        state_vec = initial_state

        for pulse in pulse_seq.pulses:
            
            H0 = gen_ham_free(basis = basis,
                  delta_L = pulse.laser_det, 
                  delta_D = k_eff*atom_veloc[v],
                  delta_R = dR,
            )

            match pulse.type_int:
                case 0:
                    Hint = gen_ham_plus(basis = basis,
                        phi_L = pulse.phase,
                        omega_R_plus = pulse.rabi_freq,
                    )
                    state_vec = time_evolve(pulse.duration, Hint + H0) @ state_vec
                case 1:
                    Hint = gen_ham_minus(basis = basis,
                        phi_L = pulse.phase,
                        omega_R_minus = pulse.rabi_freq,
                    )
                    state_vec = time_evolve(pulse.duration, Hint + H0) @ state_vec
                case 2:
                    state_vec = time_evolve(pulse.duration, H0) @ state_vec
        
        square = np.abs(state_vec)**2
        data[:,v] = square


    init_mom_dist = atom_veloc*m
    init_mom_dist_tiled = np.tile(init_mom_dist, (len(basis),1))

    basis_tiled = np.transpose(np.tile(basis, (len(atom_veloc),1)))

    final_mom_dist_tiled = init_mom_dist_tiled + (hbar*k_eff*basis_tiled)

    final_mom_dist = np.ravel(final_mom_dist_tiled)

    state_fractions = np.ravel(data)
    
    return final_mom_dist, state_fractions, init_mom_dist, rng_state




