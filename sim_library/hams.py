import numpy as np
from scipy.sparse.linalg import expm
from scipy.constants import hbar, pi
import matplotlib.pyplot as plt

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

def gen_ham_free(basis, delta_L, delta_D, delta_R):
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
    n_tot = np.size(basis)

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
    return H0

def gen_ham_plus(basis, phi_L, omega_R_plus):
    """Generates Hamiltonian for a Raman pulse in the positive direction.
    NOTE: Hamiltonians exclude factor of hbar since it is divided away when calculating time evolution.

    Args:
        basis (np.ndarray): A 1D array of the momentum basis in integer multiples hbar*k_eff.
        phi_L (float): Laser phase, specifically the phase difference between both Raman components.
        omega_R_plus (float): Rabi frequency of upwards pulse in 2pi*Hz.
    
    Returns:
        H0 (np.ndarray): A 2D array of shape (n_tot, n_tot) representing the drift Hamiltonian.
    """
    n_tot = np.size(basis)

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

def gen_ham_minus(basis, phi_L, omega_R_minus):
    """Generates Hamiltonian for a Raman pulse in the negative direction.
    NOTE: Hamiltonians exclude factor of hbar since it is divided away when calculating time evolution.

    Args:
        basis (np.ndarray): A 1D array of the momentum basis in integer multiples hbar*k_eff.
        phi_L (float): Laser phase, specifically the phase difference between both Raman components.
        omega_R_minus (float): Rabi frequency of downwards pulse in 2pi*Hz.
    
    Returns:
        H0 (np.ndarray): A 2D array of shape (n_tot, n_tot) representing the drift Hamiltonian.
    """
    n_tot = np.size(basis)

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

def time_evolve(dt, H):
    """
    Generates time evolution matrix from Hamiltonian

    Args:
        dt (float): Length of pulse/free evolution.
        Hplus (np.ndarray): A 2D array representing the Hamiltonian (divided by hbar).

    Returns:
        time_evol_mat (np.ndarray): A 2D array representing the time evolution operator for the input Hamiltonian.
    """
    time_evol_mat = expm(-1j*dt*H)
    return time_evol_mat    