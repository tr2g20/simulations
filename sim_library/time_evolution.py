import numpy as np
from scipy.sparse.linalg import expm
from scipy.constants import hbar, pi
import matplotlib.pyplot as plt
from numba import njit

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
