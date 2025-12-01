from scipy.constants import hbar, pi

m = 85.47*1.6605e-27 # Rb-85 mass
kb = 1.38064852e-23 # Boltzmann's constant
omega_2 = 2*pi*(384.230406373e12 - 1.264888e9) # Beam 2 freq.
omega_1 = 2*pi*(384.230406373e12 + 1.770843e9) # Beam 1 freq.
omega_eg = omega_1 - omega_2 # Hyperfine splitting.
k_eff = (omega_1+omega_2)/(2.998e8) # Effective wavevector.
dR = (hbar*(k_eff)**2)/(2*m) # two-photon recoil shift