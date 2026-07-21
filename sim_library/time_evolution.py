import numpy as np
from scipy.sparse.linalg import expm
from scipy.constants import hbar, pi
import matplotlib.pyplot as plt
from numba import njit

def generate_hams(n_min, n_max, phi_L, omega_R_plus, omega_R_minus, delta_L, delta_D, delta_R):
    """Generates Hamiltonians necessary for simulating alternating pulse sequences.
    NOTE: Hamiltonians exclude factor of hbar since it is divided away when calculating time evolution.

    Args:
        n_min (int): Lowest momentum state (i.e. bottom of the ladder) in units of hbar*k_eff.
        n_max (int): Highest momentum state (i.e top of the ladder) in units of hbar*k_eff. Basis then runs from n_min to n_max in integer steps.
        phi_L (float): Laser phase, specifically the phase difference between both Raman components.
        omega_R_plus (float): Rabi frequency of upwards pulse in 2pi*Hz.
        omega_R_minus (float): Rabi frequency of downwards pulse in 2pi*Hz.
        delta_L (float): Two-photon detuning of laser in 2pi*Hz.
        delta_D (float): Doppler shift detuning in 2pi*Hz.
        delta_R (float): Recoil shift detuning in 2pi*Hz.
        

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - H0 (np.ndarray): A 2D array of shape (n_max-n_min, n_max-n_min) representing the drift Hamiltonian.
            - Hplus (np.ndarray): A 2D array of shape (n_max-n_min, n_max-n_min) representing the Hamiltonian for an upwards Raman pulse.
            - Hminus (np.ndarray): A 2D array of shape (n_max-n_min, n_max-n_min) representing the Hamiltonian for a downwards Raman pulse.
            - basis (np.ndarray): A 1D array containing the basis of momentum states in units of hbar*k_eff.
    """
    n_tot = n_max - n_min + 1
    basis = np.arange(n_min, n_max + 1)
    
    ### Generate H0 ###
    # Diagonal matrix
    # Ground states are even momenta and excited states are odd
    H0 = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for i in range(n_tot):
        n = basis[i]
        if n % 2 == 0:
            H0[i,i] = n*(delta_D + n*delta_R)
        else:
            H0[i,i] = n*(delta_D + n*delta_R) - delta_L
    
    ### Generate H+ ###
    # Care is taken to not go out of range of array, 
    # for Hplus if ground (even mom) then n < n_max,
    # if excited (odd mom) then n > n_min
    Hplus = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for row in range(n_tot):
        n = basis[row]
        if (n % 2 == 0) and (n < n_max):
            Hplus[row, row + 1] = (omega_R_plus/2)*np.exp(-1j*phi_L)
        elif (n % 2 != 0) and (n > n_min):
            Hplus[row, row - 1] = (omega_R_plus/2)*np.exp(1j*phi_L)
            
    ### Generate H- ###
    # Care is taken to not go out of range of array, 
    # for Hplus if ground (even mom) then n > n_min,
    # if excited (odd mom) then n < n_max
    Hminus = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for row in range(n_tot):
        n = basis[row]
        if (n % 2 == 0) and (n > n_min):
            Hminus[row, row - 1] = (omega_R_minus/2)*np.exp(-1j*phi_L)
        elif (n % 2 != 0) and (n < n_max):
            Hminus[row, row + 1] = (omega_R_minus/2)*np.exp(1j*phi_L)
            
    return H0, Hplus, Hminus, basis

@njit
def gen_ham_free(basis: np.ndarray, delta_L: float, delta_D: float, delta_R: float):
    """Generates Hamiltonian for free evolution.
    NOTE: Hamiltonians exclude factor of hbar since it is divided away when calculating time evolution.

    Args:
        basis (np.ndarray): A 1D array of the momentum basis in integer multiples hbar*k_eff.
        delta_L (float): Two-photon detuning of laser in 2pi*Hz.
        delta_D (float): Doppler shift detuning in 2pi*Hz.
        delta_R (float): Recoil shift detuning in 2pi*Hz.
    
    Returns:
        H0 (np.ndarray): A 2D array of shape (n_tot, n_tot) representing the drift Hamiltonian.
    """
    n_tot = len(basis)

    ### Generate H0 ###
    # Diagonal matrix
    # Ground states are even momenta and excited states are odd
    H0 = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for i in range(n_tot):
        n = basis[i]
        if n % 2 == 0:
            H0[i,i] = -n*(delta_D + n*delta_R)
        else:
            H0[i,i] = -n*(delta_D + n*delta_R) + delta_L
    return H0

@njit
def gen_ham_plus(basis: np.ndarray, phi_L: float, omega_R_plus: float):
    """Generates Hamiltonian for a Raman pulse in the positive direction.
    NOTE: Hamiltonians exclude factor of hbar since it is divided away when calculating time evolution.

    Args:
        basis (np.ndarray): A 1D array of the momentum basis in integer multiples hbar*k_eff.
        phi_L (float): Laser phase, specifically the phase difference between both Raman components.
        omega_R_plus (float): Rabi frequency of upwards pulse in 2pi*Hz.
    
    Returns:
        Hplus (np.ndarray): A 2D array of shape (n_tot, n_tot) representing the Hamiltonian for an upwards Raman pulse.
    """
    n_tot = len(basis)

    ### Generate H+ ###
    # Care is taken to not go out of range of array, 
    # for Hplus if ground (even mom) then n < n_max,
    # if excited (odd mom) then n > n_min
    Hplus = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for row in range(n_tot):
        n = basis[row]
        if (n % 2 == 0) and (n < basis[-1]):
            Hplus[row, row + 1] = (omega_R_plus/2)*np.exp(-1j*phi_L)
        elif (n % 2 != 0) and (n > basis[0]):
            Hplus[row, row - 1] = (omega_R_plus/2)*np.exp(1j*phi_L)
    return Hplus

@njit
def gen_ham_minus(basis: np.ndarray, phi_L: float, omega_R_minus: float):
    """Generates Hamiltonian for a Raman pulse in the negative direction.
    NOTE: Hamiltonians exclude factor of hbar since it is divided away when calculating time evolution.

    Args:
        basis (np.ndarray): A 1D array of the momentum basis in integer multiples hbar*k_eff.
        phi_L (float): Laser phase, specifically the phase difference between both Raman components.
        omega_R_minus (float): Rabi frequency of downwards pulse in 2pi*Hz.
    
    Returns:
        Hminus (np.ndarray): A 2D array of shape (n_tot, n_tot) representing the Hamiltonian for a downwards Raman pulse.
    """
    n_tot = len(basis)

    ### Generate H- ###
    # Care is taken to not go out of range of array, 
    # for Hplus if ground (even mom) then n > n_min,
    # if excited (odd mom) then n < n_max
    Hminus = np.zeros((n_tot, n_tot), dtype=np.complex128)
    for row in range(n_tot):
        n = basis[row]
        if (n % 2 == 0) and (n > basis[0]):
            Hminus[row, row - 1] = (omega_R_minus/2)*np.exp(-1j*phi_L)
        elif (n % 2 != 0) and (n < basis[-1]):
            Hminus[row, row + 1] = (omega_R_minus/2)*np.exp(1j*phi_L)
    return Hminus

@njit
def time_evolve(state_vec: np.ndarray, dt: float, H: np.ndarray):
    """
    Calculates the time-evolved state vector using the given Hamiltonian.

    Args:
        state_vec (np.ndarray): A 1D complex array representing the initial state vector.
        dt (float): Length of pulse/free evolution.
        H (np.ndarray): A 2D array representing the total Hamiltonian (divided by hbar).

    Returns:
        new_state_vec (np.ndarray): A 1D complex array representing the updated state vector after time evolution.
    """
    eig_vals, eig_vecs = np.linalg.eigh(H)
    
    # Brackets to ensure correct multiplication order such that each step is matrix x vector or vector x vector
    new_state_vec = eig_vecs @ (np.exp(-1j * dt * eig_vals) * (eig_vecs.conj().T @ state_vec))
 
    return new_state_vec    

def time_evolve_old(dt, H):
    """
    Generates time evolution matrix from Hamiltonian

    Args:
        dt (float): Length of pulse/free evolution.
        H (np.ndarray): A 2D array representing the total Hamiltonian (divided by hbar).

    Returns:
        time_evol_mat (np.ndarray): A 2D array representing the time evolution operator for the input Hamiltonian.
    """
    time_evol_mat = expm(-1j*dt*H)
    return time_evol_mat    

@njit
def evolve_free(dt: float, state_vec: np.ndarray, delta_D: float, delta_R: float, delta_L: float, basis: np.ndarray):
    """
    Evolves wavefunction over a time interval dt using analytical expressions for time-dependent state amplitudes during free evolution. 
    Parameters are constant over this time interval.

    Args:
        dt (float): Length of pulse/free evolution.
        state_vec (np.ndarray): The initial momentum state vector (elements are np.complex128). 
        delta_D (float): Doppler shift detuning in 2pi*Hz.
        delta_R (float): Recoil shift detuning in 2pi*Hz.
        delta_L (float): Two-photon detuning of laser in 2pi*Hz.
        basis (np.ndarray): A 1D array of the momentum basis in integer multiples hbar*k_eff.

    Returns:
        new_state_vec (np.ndarray): Time-evolved state vector.
    """
    new_state_vec = np.zeros_like(state_vec)
    for i, n in enumerate(basis):
        if n % 2 == 0:
            new_state_vec[i] = state_vec[i]*np.exp(1j*n*(delta_D+(n*delta_R))*dt)
        else:
            new_state_vec[i] = state_vec[i]*np.exp(1j*(n*(delta_D+(n*delta_R))-delta_L)*dt)
    return new_state_vec

@njit
def evolve_uppulse(dt: float, state_vec: np.ndarray, omega_R_plus: float, phi_L: float, delta_D: float, delta_R: float, delta_L: float, basis: np.ndarray):
    """
    Evolves wavefunction over a time interval dt using analytical expressions for time-dependent state amplitudes during an upwards pulse. 
    Parameters are constant over this time interval.

    Args:
        dt (float): Length of pulse/free evolution.
        state_vec (np.ndarray): The initial momentum state vector (elements are np.complex128). 
        omega_R_plus (float): Rabi frequency of upwards pulse in 2pi*Hz.
        phi_L (float): Laser phase, specifically the phase difference between both Raman components.
        delta_D (float): Doppler shift detuning in 2pi*Hz.
        delta_R (float): Recoil shift detuning in 2pi*Hz.
        delta_L (float): Two-photon detuning of laser in 2pi*Hz.
        basis (np.ndarray): A 1D array of the momentum basis in integer multiples hbar*k_eff.

    Returns:
        new_state_vec (np.ndarray): Time-evolved state vector.
    """
    new_state_vec = np.zeros_like(state_vec)
    start = 0
    end = len(basis)-1
    if basis[0] % 2 == 1:
        new_state_vec[0] = state_vec[0]*np.exp(1j*(basis[0]*(delta_D+(basis[0]*delta_R))-delta_L)*dt)
        start = 1
    if basis[-1] % 2 == 0:
        new_state_vec[-1] = state_vec[-1]*np.exp(1j*(basis[-1]*(delta_D+(basis[-1]*delta_R)))*dt)
    for i in range(start,end,2):
        n = basis[i]
        delta_p = delta_D + (2*n+1)*delta_R - delta_L
        omega_eff = np.sqrt((omega_R_plus**2)+(delta_p**2))
        P = np.exp(1j*(n*(delta_D+(n*delta_R))+(delta_p/2))*dt)
        C = np.cos(omega_eff*dt/2)-(1j*(delta_p/omega_eff)*np.sin(omega_eff*dt/2))
        C_conj = np.cos(omega_eff*dt/2)+(1j*(delta_p/omega_eff)*np.sin(omega_eff*dt/2))
        S = np.exp(1j*phi_L)*(omega_R_plus/omega_eff)*np.sin(omega_eff*dt/2)
        S_conj = np.exp(-1j*phi_L)*(omega_R_plus/omega_eff)*np.sin(omega_eff*dt/2)
        new_state_vec[i] = P*((C*state_vec[i])-(1j*S_conj*state_vec[i+1]))
        new_state_vec[i+1] = P*((C_conj*state_vec[i+1])-(1j*S*state_vec[i]))

    return new_state_vec

@njit
def evolve_downpulse(dt: float, state_vec: np.ndarray, omega_R_minus: float, phi_L: float, delta_D: float, delta_R: float, delta_L: float, basis: np.ndarray):
    """
    Evolves wavefunction over a time interval dt using analytical expressions for time-dependent state amplitudes during an upwards pulse. 
    Parameters are constant over this time interval.

    Args:
        dt (float): Length of pulse/free evolution.
        state_vec (np.ndarray): The initial momentum state vector (elements are np.complex128). 
        omega_R_minus (float): Rabi frequency of minus pulse in 2pi*Hz.
        phi_L (float): Laser phase, specifically the phase difference between both Raman components.
        delta_D (float): Doppler shift detuning in 2pi*Hz.
        delta_R (float): Recoil shift detuning in 2pi*Hz.
        delta_L (float): Two-photon detuning of laser in 2pi*Hz.
        basis (np.ndarray): A 1D array of the momentum basis in integer multiples hbar*k_eff.

    Returns:
        new_state_vec (np.ndarray): Time-evolved state vector.
    """
    new_state_vec = np.zeros_like(state_vec)
    start = 0
    end = len(basis)-1
    if basis[0] % 2 == 0:
        new_state_vec[0] = state_vec[0]*np.exp(1j*(basis[0]*(delta_D+(basis[0]*delta_R)))*dt)
        start = 1
    if basis[-1] % 2 == 1:
        new_state_vec[-1] = state_vec[-1]*np.exp(1j*(basis[-1]*(delta_D+(basis[-1]*delta_R))-delta_L)*dt)
    for i in range(start,end,2):
        n = basis[i]
        delta_p = delta_D + (2*n+1)*delta_R - delta_L
        delta_m = delta_D + (2*n+1)*delta_R + delta_L
        omega_eff = np.sqrt((omega_R_minus**2)+(delta_m**2))
        P = np.exp(1j*(n*(delta_D+(n*delta_R))+(delta_p/2))*dt)
        C = np.cos(omega_eff*dt/2)-(1j*(delta_m/omega_eff)*np.sin(omega_eff*dt/2))
        C_conj = np.cos(omega_eff*dt/2)+(1j*(delta_m/omega_eff)*np.sin(omega_eff*dt/2))
        S = np.exp(-1j*phi_L)*(omega_R_minus/omega_eff)*np.sin(omega_eff*dt/2)
        S_conj = np.exp(1j*phi_L)*(omega_R_minus/omega_eff)*np.sin(omega_eff*dt/2)
        new_state_vec[i] = P*((C*state_vec[i])-(1j*S_conj*state_vec[i+1]))
        new_state_vec[i+1] = P*((C_conj*state_vec[i+1])-(1j*S*state_vec[i]))

    return new_state_vec
