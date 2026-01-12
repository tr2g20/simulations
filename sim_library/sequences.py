import numpy as np
from scipy.constants import pi
from sim_library.constants import dR
from sim_library.hams import gen_ham_free, gen_ham_plus, gen_ham_minus

class Pulse:
    """
    Base class representing one of the three fundamental operations of the MSQC (up pulse, down pulse, free evolution).

    Attributes:
        laser_det (float or np.ndarray): Two-photon detuning of laser in 2pi*Hz.
        duration (float): Length of pulse/free evolution in seconds.
    """
    laser_det: np.ndarray
    duration: float
    def __init__(self, laser_det, duration):
        self.laser_det = laser_det
        self.duration = duration

    def gen_ham_arr(self, basis, doppler_shift):
        raise NotImplementedError("Subclasses must implement gen_ham_arr")


class UpPulse(Pulse):
    """
    Represents a Raman pulse in the positive direction.

    Inherits from Pulse

    The phase, rabi_freq and laser_det attributes are arrays of values, where that value is held constant
    for a certain time interval. The time interval is equal for each value in the array and is determined from
    the duration divided by the total number of values. The phase, rabi_freq and laser_det as functions of time
    will look like steps, since the way things are simulated the Hamiltonian needs to be constant at each discrete 
    time interval.
    If you have a phase profile that you want to input that has one phase value per each time value (instead of per time interval) 
    you will need to remove either the first or last data point depending on how the pulse is defined.

    Attributes:
        phase (float or np.ndarray): Laser phase, specifically the phase difference between both Raman components.
        rabi_freq (float or np.ndarray): Rabi frequency of pulse in 2pi*Hz.
        type (str): Identifier string, fixed as 'up'.
        type_int (int): Integer identifier, fixed as 0.
    """
    phase: np.ndarray
    rabi_freq: np.ndarray
    type: str
    type_int: int
    def __init__(self, laser_det, phase, rabi_freq, duration):
        """
        Raises:
            ValueError: If `phase`, `rabi_freq`, and `laser_det` shapes do not match.
        """
        if not (phase.shape == rabi_freq.shape == laser_det.shape):
            raise ValueError(
                f"Array dimensions must match. Received shapes: "
                f"phase {phase.shape}, "
                f"rabi_freq {rabi_freq.shape}, "
                f"laser_det {laser_det.shape}"
            )
        
        super().__init__(laser_det, duration)
        self.phase = phase
        self.rabi_freq = rabi_freq
        self.type = 'up'
        self.type_int = 0

    def gen_ham_list(self, basis, doppler_shift):
        """
        Generates list of Hamiltonians at each time interval of the pulse
        H = H0 + Hplus
        Chosen to be a list of ndarrays instead of a singular ndarray to make concatenation easier with other pulses.
        
        Args:
            basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
            doppler_shift (float): Doppler shift detuning in 2pi*Hz.

        Returns:
            list[np.ndarray]: A list of 2D arrays where each element of the list is the pulse Hamiltonian at a specific time interval.
        """
        hams_list = []
        for i in range(len(self.phase)):
            H0 = gen_ham_free(basis = basis,
                  delta_L = self.laser_det[i], 
                  delta_D = doppler_shift,
                  delta_R = dR,
            )
            Hplus = gen_ham_plus(basis = basis,
                        phi_L = self.phase[i],
                        omega_R_plus = self.rabi_freq[i],
                    )
            hams_list.append(H0 + Hplus)
        return hams_list



class DownPulse(Pulse):
    """
    Represents a Raman pulse in the negative direction.

    Inherits from Pulse

    The phase, rabi_freq and laser_det attributes are arrays of values, where that value is held constant
    for a certain time interval. The time interval is equal for each value in the array and is determined from
    the duration divided by the total number of values. The phase, rabi_freq and laser_det as functions of time
    will look like steps, since the way things are simulated the Hamiltonian needs to be constant at each discrete 
    time interval.
    If you have a phase profile that you want to input that has one phase value per each time value (instead of per time interval) 
    you will need to remove either the first or last data point depending on how the pulse is defined.

    Attributes:
        phase (float or np.ndarray): Laser phase, specifically the phase difference between both Raman components.
        rabi_freq (float or np.ndarray): Rabi frequency of pulse in 2pi*Hz.
        type (str): Identifier string, fixed as 'down'.
        type_int (int): Integer identifier, fixed as 1.
    """
    phase: np.ndarray
    rabi_freq: np.ndarray
    type: str
    type_int: int
    def __init__(self, laser_det, phase, rabi_freq, duration):
        """
        Raises:
            ValueError: If `phase`, `rabi_freq`, and `laser_det` shapes do not match.
        """
        if not (phase.shape == rabi_freq.shape == laser_det.shape):
            raise ValueError(
                f"Array dimensions must match. Received shapes: "
                f"phase {phase.shape}, "
                f"rabi_freq {rabi_freq.shape}, "
                f"laser_det {laser_det.shape}"
            )
        
        super().__init__(laser_det, duration)
        self.phase = phase
        self.rabi_freq = rabi_freq
        self.type = 'down'
        self.type_int = 1

    def gen_ham_list(self, basis, doppler_shift):
        """
        Generates list of Hamiltonians at each time interval of the pulse
        H = H0 + Hplus
        Chosen to be a list of ndarrays instead of a singular ndarray to make concatenation easier with other pulses.
        
        Args:
            basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
            doppler_shift (float): Doppler shift detuning in 2pi*Hz.

        Returns:
            list[np.ndarray]: A list of 2D arrays where each element of the list is the pulse Hamiltonian at a specific time interval.
        """
        hams_list = []
        for i in range(len(self.phase)):
            H0 = gen_ham_free(basis = basis,
                  delta_L = self.laser_det[i], 
                  delta_D = doppler_shift,
                  delta_R = dR,
            )
            Hminus = gen_ham_minus(basis = basis,
                        phi_L = self.phase[i],
                        omega_R_minus = self.rabi_freq[i],
                    )
            hams_list.append(H0 + Hminus)
        return hams_list

class FreeEvolution(Pulse):
    """
    Represents a period of free evolution.

    Inherits from Pulse

    The phase, rabi_freq and laser_det attributes are arrays of values, where that value is held constant
    for a certain time interval. The time interval is equal for each value in the array and is determined from
    the duration divided by the total number of values. The phase, rabi_freq and laser_det as functions of time
    will look like steps, since the way things are simulated the Hamiltonian needs to be constant at each discrete 
    time interval.
    If you have a phase profile that you want to input that has one phase value per each time value (instead of per time interval) 
    you will need to remove either the first or last data point depending on how the pulse is defined.

    Attributes:
        type (str): Identifier string, fixed as 'free'.
        type_int (int): Integer identifier, fixed as 2.
    """
    type: str
    type_int: int
    def __init__(self, laser_det, duration):
        super().__init__(laser_det, duration)
        self.type = 'free'
        self.type_int = 2

    def gen_ham_list(self, basis, doppler_shift):
        """
        Generates list of Hamiltonians at each time interval of the pulse
        H = H0 + Hplus
        Chosen to be a list of ndarrays instead of a singular ndarray to make concatenation easier with other pulses.
        
        Args:
            basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
            doppler_shift (float): Doppler shift detuning in 2pi*Hz.

        Returns:
            list[np.ndarray]: A list of 2D arrays where each element of the list is the pulse Hamiltonian at a specific time interval.
        """
        hams_list = []
        for i in range(len(self.laser_det)):
            H0 = gen_ham_free(basis = basis,
                  delta_L = self.laser_det[i], 
                  delta_D = doppler_shift,
                  delta_R = dR,
            )
            hams_list.append(H0)
        return hams_list

class PulseSequence:
    """
    Represents an ordered sequence of Pulse objects.

    times will have a length that is 1 longer than hams. Since the discrete hamiltonians segments are defined
    as being constant during each timer interval, instead of being defined at each time step.

    Note: times and hams are unpopulated until gen_hams is run.

    Attributes:
        pulses (list[Pulse]): Ordered list of Pulse objects.
        times (nump.ndarray): A 1D array of each time step (in seconds) for the whole pulse sequence.
        hams (np.ndarray): A 3D array of the Hamiltonians at each time interval for the whole pulse sequence.
    """

    pulses: list
    times: np.ndarray
    # rabis: np.ndarray
    # detunings: np.ndarray
    # phases: np.ndarray
    hams: np.ndarray

    def __init__(self):
        self.pulses = []

    def add_pulse(self, pulse_object):
        """
        Adds a single Pulse object to the end of the sequence.

        Args:
            pulse_object (PulseSubclass): An instance of Pulse, UpPulse, DownPulse, or FreeEvolution.

        Raises:
            TypeError: If the object added is not a Pulse object.
        """
        if isinstance(pulse_object, Pulse):
            self.pulses.append(pulse_object)
        else:
            raise TypeError("Only Pulse objects (or subclasses) can be added.")
    
    def get_n_steps(self):
        """
        Calculates the number of time steps in the whole pulse sequence.
        
        Returns:
            int: Integer number of time steps in pulse sequence (or number of time intervals + 1).
        """
        count = 1
        for pulse in self.pulses:
            count += len(pulse.laser_det)
        return count
    
    def gen_hams(self, basis, doppler_shift):
        """
        Iterates through each pulse and concatenates all the Hamiltonians to create a single Hamiltonian (as a function of time) for the entire pulse sequence.
        Saves pulse sequence hamiltonian in hams. Also updates times to contain the time steps for the whole sequence.

        Args:
            basis (numpy.ndarray): The 1D array of momentum basis states (in integers of hbar*k_eff).
            doppler_shift (float): Doppler shift detuning in 2pi*Hz.

        Returns:
            None: Updates times and hams attributes (I may get rid of these variables and change it to just return them).
        """
        # rabis = np.concatenate([[0], pulse.rabi_freq[1:] for pulse in self.pulses])
        # detunings = np.concatenate([[0], pulse.laser_det[1:] for pulse in self.pulses])
        # phases = np.concatenate([[0], pulse.phase[1:] for pulse in self.pulses])
        times_list = [np.array([0])]
        hams_list = []
        for pulse in self.pulses:
            pulse_n_steps = len(pulse.laser_det)
            times_list.append(np.linspace(times_list[-1][-1], times_list[-1][-1] + pulse.duration, pulse_n_steps + 1)[1:])
            hams_list += pulse.gen_ham_list(basis=basis, doppler_shift=doppler_shift)
        self.times = np.concatenate(times_list)
        self.hams = np.array(hams_list)           
        
def gen_resonant_pm_seq(no_pulses: int, rabi_freq: float, n_steps: int, p_start: int = 0, dir: str = 'pos'):
    """
    Generates sequence of rectangular pi pulses of alternating direction, with no free evolution. 
    Rabi frequency stays constant for each pulse and phase is set to 0.
    Laser detuning is set to be resonant for each individual pulse.

    This code is quite janky, I could make it a more readable method but it works so I probably wont change it.

    Args:
        no_pulses (int): The total number of pulses.
        rabi_freq (float or numpy.ndarray): Rabi frequency of all pulses in 2pi*Hz.
        n_steps (int): The number of time steps each pulse is split into.
        p_start (int): The starting momentum for the pulse sequence to target.
        dir (str): Direction of total momentum change, 'pos' or 'neg'
                                            
    Returns:
        PulseSequence: An object containing the generated sequence of alternating pulses.
    """
    pulse_seq = PulseSequence()

    phases = np.full(n_steps, 0)
    rabi_frequencies = np.full(n_steps, rabi_freq)

    if dir == 'pos':
        shift = 0
    else:
        shift = 1

    p = p_start - shift

    for n in range(shift, no_pulses + shift):
        if n % 2 == 0:
            detunings = np.full(n_steps, (2*p + 1)*dR)
            pulse_seq.add_pulse(UpPulse(laser_det=detunings, phase=phases, rabi_freq=rabi_frequencies, duration=pi/rabi_freq))
        else:
            detunings = np.full(n_steps, -1*(2*p + 1)*dR)
            pulse_seq.add_pulse(DownPulse(laser_det=detunings, phase=phases, rabi_freq=rabi_frequencies, duration=pi/rabi_freq))
        p += (-1)**shift

    return pulse_seq

def gen_offresonant_pm_seq(no_pulses: int, rabi_freq: float, n_steps: int, p_start: int = 0, dir: str = 'pos'):
    """
    Generates sequence of rectangular pi pulses of alternating direction, with no free evolution. 
    Rabi frequency stays constant for each pulse and phase is set to 0.
    Laser detuning is set to be resonant with only the first pulse and constant throughout.

    This code is quite janky, I could make it a more readable method but it works so I probably wont change it.

    Args:
        no_pulses (int): The total number of pulses.
        rabi_freq (float or numpy.ndarray): Rabi frequency of all pulses in 2pi*Hz.
        n_steps (int): The number of time steps each pulse is split into.
        p_start (int): The starting momentum for the pulse sequence to target.
        dir (str): Direction of total momentum change, 'pos' or 'neg'
                                            
    Returns:
        PulseSequence: An object containing the generated sequence of alternating pulses.
    """
    pulse_seq = PulseSequence()


    phases = np.full(n_steps, 0)
    rabi_frequencies = np.full(n_steps, rabi_freq)

    if dir == 'pos':
        shift = 0
    else:
        shift = 1

    p = p_start - shift

    if shift % 2 == 0:
        detunings = np.full(n_steps, (2*p + 1)*dR)
    else:
        detunings = np.full(n_steps, -1*(2*p + 1)*dR)

    for n in range(shift, no_pulses + shift):
        if n % 2 == 0:
            pulse_seq.add_pulse(UpPulse(laser_det=detunings, phase=phases, rabi_freq=rabi_frequencies, duration=pi/rabi_freq))
        else:
            pulse_seq.add_pulse(DownPulse(laser_det=detunings, phase=phases, rabi_freq=rabi_frequencies, duration=pi/rabi_freq))
    
    return pulse_seq

def gen_resonant_pm_seq_fast(no_pulses: int, rabi_freq: float, n_steps: int, p_start: int = 0, dir: str = 'pos'):
    pulse_seq = gen_resonant_pm_seq(no_pulses=no_pulses, rabi_freq=rabi_freq, n_steps=n_steps, p_start=p_start, dir=dir)
    pulse_seq_fast = gen_resonant_pm_seq(no_pulses=no_pulses, rabi_freq=rabi_freq, n_steps=1, p_start=p_start, dir=dir)
    return pulse_seq, pulse_seq_fast

def gen_offresonant_pm_seq_fast(no_pulses: int, rabi_freq: float, n_steps: int, p_start: int = 0, dir: str = 'pos'):
    pulse_seq = gen_offresonant_pm_seq(no_pulses=no_pulses, rabi_freq=rabi_freq, n_steps=n_steps, p_start=p_start, dir=dir)
    pulse_seq_fast = gen_offresonant_pm_seq(no_pulses=no_pulses, rabi_freq=rabi_freq, n_steps=1, p_start=p_start, dir=dir)
    return pulse_seq, pulse_seq_fast