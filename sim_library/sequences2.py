import numpy as np
from scipy.constants import pi
from sim_library.constants import dR, omega_eg
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

    def add_pulses(self, pulse_object):
        """
        Appends Pulses to the PulseSequence.

        Args:
            pulse_object (Pulse | PulseSequence | list[Pulse | PulseSequence]): Can be a Pulse, a PulseSequence or a list of either.

        Raises:
            TypeError: If the object added is not a Pulse or PulseSequence object.
        """
        if isinstance(pulse_object, Pulse):
            self.pulses.append(pulse_object)
        elif isinstance(pulse_object, list):
            for pulse in pulse_object:
                self.add_pulses(pulse)
        elif isinstance(pulse_object, PulseSequence):
            for pulse in pulse_object.pulses:
                self.pulses.append(pulse)
        else:
            raise TypeError("Only Pulse or PulseSequence objects can be added.")
    
    def list_pulses(self):
        """
        Prints the type of each pulse as a string
        """
        for pulse in self.pulses:
            print(pulse.type)

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
            pulse_seq.add_pulses(UpPulse(laser_det=detunings, phase=phases, rabi_freq=rabi_frequencies, duration=pi/rabi_freq))
        else:
            detunings = np.full(n_steps, -1*(2*p + 1)*dR)
            pulse_seq.add_pulses(DownPulse(laser_det=detunings, phase=phases, rabi_freq=rabi_frequencies, duration=pi/rabi_freq))
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
            pulse_seq.add_pulses(UpPulse(laser_det=detunings, phase=phases, rabi_freq=rabi_frequencies, duration=pi/rabi_freq))
        else:
            pulse_seq.add_pulses(DownPulse(laser_det=detunings, phase=phases, rabi_freq=rabi_frequencies, duration=pi/rabi_freq))
    
    return pulse_seq

def gen_resonant_pm_seq_fast(no_pulses: int, rabi_freq: float, n_steps: int, p_start: int = 0, dir: str = 'pos'):
    """
    Generates PulseSequence based on parameters and also a copy of this that only has one time step per pulse. 
    This useful for when you want to plot the state trajectories and the initial and final momentum distribution
    at the same time without taking as long to simulate each intermediate momentum distribution.
    """
    pulse_seq = gen_resonant_pm_seq(no_pulses=no_pulses, rabi_freq=rabi_freq, n_steps=n_steps, p_start=p_start, dir=dir)
    pulse_seq_fast = gen_resonant_pm_seq(no_pulses=no_pulses, rabi_freq=rabi_freq, n_steps=1, p_start=p_start, dir=dir)
    return pulse_seq, pulse_seq_fast

def gen_offresonant_pm_seq_fast(no_pulses: int, rabi_freq: float, n_steps: int, p_start: int = 0, dir: str = 'pos'):
    """
    Same as gen_resonant_pm_seq_fast but for an off resonant sequence.
    """
    pulse_seq = gen_offresonant_pm_seq(no_pulses=no_pulses, rabi_freq=rabi_freq, n_steps=n_steps, p_start=p_start, dir=dir)
    pulse_seq_fast = gen_offresonant_pm_seq(no_pulses=no_pulses, rabi_freq=rabi_freq, n_steps=1, p_start=p_start, dir=dir)
    return pulse_seq, pulse_seq_fast

def gen_MDFE_seq(free_time: float, rabi_freq: float, time_steps: int, detuning: float = dR):
    """
    Generates a Momentum-Dependent Free Evolution (MDFE) pulse sequence of rectangular pulses.

    Args:
        free_time (float): Total duration of the free evolution.
        rabi_freq (float): The Rabi frequency in 2*pi Hz.
        time_steps (int): Number of discrete time steps per pulse or freevolution.
        detuning (float): Laser detuning of up and down pulses. Defaults to global `dR`.

    Returns:
        PulseSequence: PulseSequence object containing the MDFE sequence.
    """
    mom_dependent_free_evolution = PulseSequence()

    rabi_time = 2*pi/rabi_freq

    up_pulse = UpPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/2)
    down_pulse = DownPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/2)
    freevolve = FreeEvolution(laser_det=np.full(time_steps, dR), duration=free_time/4)

    mom_dependent_free_evolution.add_pulses([freevolve, up_pulse, freevolve, up_pulse, freevolve, down_pulse, freevolve, down_pulse])
    return mom_dependent_free_evolution

def not0_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    """
    Generates a pulse sequence for a NOT gate acting on first qubit

    Args:
        rabi_freq (float): The Rabi frequency in 2*pi Hz.
        time_steps (int): Number of discrete time steps per pulse or freevolution.
        detuning (float): Magnitude of laser detuning during free evolution. Default is half hyperfine splitting. 
                            If changing this only input a positive value to avoid changing rotation direction.

    Returns:
        PulseSequence: PulseSequence object containing the NOT gate sequence.
    """
    rabi_time = 2*pi/rabi_freq
    free_time = 2*pi/detuning 

    not_gate = PulseSequence()

    freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=free_time/4) #pi/2
    pulse_1 = UpPulse(laser_det=np.full(time_steps, dR), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/2) #up pi pulse, phase = 0

    not_gate.add_pulses([freevolve_1, 
                        pulse_1, 
                        freevolve_1])
    
    return not_gate

def exchange10_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    rabi_time = 2*pi/rabi_freq
    free_time = 2*pi/detuning

    exchange10 = PulseSequence()

    pulse_1 = DownPulse(laser_det=np.full(time_steps, dR), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4) #down pi/2 pulse, phase = pi/2

    MDFE_1 = gen_MDFE_seq(free_time=pi/(4*dR), rabi_freq=rabi_freq, time_steps=time_steps) # pi/4
    freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, +detuning)), duration=5*free_time/8) #5pi/4 
    freevolve_2 = FreeEvolution(laser_det=(np.full(time_steps, +detuning)), duration=free_time/2) #pi

    exchange10.add_pulses([pulse_1,
                    freevolve_1,
                    MDFE_1,
                    pulse_1,
                    freevolve_2])
    
    return exchange10

def anti_CNOT10_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    rabi_time = 2*pi/rabi_freq
    free_time = 2*pi/detuning

    antiCNOT = PulseSequence()

    pulse_1 = UpPulse(laser_det=np.full(time_steps, dR), phase=np.full(time_steps, pi/2), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4) #up pi/2 pulse, phase = pi/2
    pulse_2 = DownPulse(laser_det=np.full(time_steps, dR), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time) #down 2pi pulse, phase = 0
    MDFE_1 = gen_MDFE_seq(free_time=pi/(4*dR), rabi_freq=rabi_freq, time_steps=time_steps) #pi/4
    freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=7*free_time/8) #7pi/4
    freevolve_2 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=free_time/2) #pi


    antiCNOT.add_pulses([pulse_1,
                    freevolve_1,
                    MDFE_1,
                    pulse_1,
                    freevolve_2,
                    pulse_2])
    
    return antiCNOT

def swap23_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    rabi_time = 2*pi/rabi_freq
    free_time = 2*pi/detuning

    swap23 = PulseSequence()

    pulse_1 = UpPulse(laser_det=np.full(time_steps, dR), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4) #up pi/2 pulse, phase = pi/2
    MDFE_1 = gen_MDFE_seq(free_time=pi/(8*dR), rabi_freq=rabi_freq, time_steps=time_steps) #pi/8
    freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=7*free_time/16) #7pi/8
    freevolve_2 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=3*free_time/16) #3pi/8

    swap23.add_pulses([pulse_1,
                        freevolve_1,
                        MDFE_1,
                        pulse_1,
                        freevolve_2,
                        MDFE_1,
                        pulse_1,
                        freevolve_1,
                        MDFE_1,
                        pulse_1
                        ])
    
    return swap23

def swap34_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    rabi_time = 2*pi/rabi_freq
    free_time = 2*pi/detuning

    swap34 = PulseSequence()

    pulse_1 = DownPulse(laser_det=np.full(time_steps, dR), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4) #down pi/2 pulse, phase = pi/2
    MDFE_1 = gen_MDFE_seq(free_time=pi/(8*dR), rabi_freq=rabi_freq, time_steps=time_steps) #pi/8
    freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, +detuning)), duration=5*free_time/16) #5pi/8
    freevolve_2 = FreeEvolution(laser_det=(np.full(time_steps, +detuning)), duration=free_time/16) #pi/8 

    swap34.add_pulses([pulse_1,
                        freevolve_1,
                        MDFE_1,
                        pulse_1,
                        freevolve_2,
                        MDFE_1,
                        pulse_1,
                        freevolve_1,
                        MDFE_1,
                        pulse_1,
                        ])
    
    return swap34


def ramsey_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    rabi_time = 2*pi/rabi_freq
    free_time = 2*pi/detuning 

    ramsey = PulseSequence()

    pulse_1 = UpPulse(laser_det=np.full(time_steps, dR), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4) #up pi/2 pulse, phase = pi/2
    MDFE_1 = gen_MDFE_seq(free_time=pi/(8*dR), rabi_freq=rabi_freq, time_steps=time_steps) #pi/8
    freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=3*free_time/16) #3pi/8

    ramsey.add_pulses([pulse_1,
                    freevolve_1,
                    MDFE_1,
                    pulse_1,
                    ])
    
    return ramsey

def swap45_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    free_time = 2*pi/detuning 

    swap45 = PulseSequence()

    ramsey = ramsey_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    MDFE_1 = gen_MDFE_seq(free_time=pi/(8*dR), rabi_freq=rabi_freq, time_steps=time_steps) #pi/8
    freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=15*free_time/16) #15pi/8

    swap45.add_pulses([ramsey,
                    freevolve_1,
                    MDFE_1,
                    ramsey,
                    ])
    
    return swap45

def exchange21_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    rabi_time = 2*pi/rabi_freq
    free_time = 2*pi/detuning 

    exchange21 = PulseSequence()

    not_gate = not0_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    ex10 = exchange10_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    antiCNOT = anti_CNOT10_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    swap23 = swap23_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    swap34 = swap34_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    swap45 = swap45_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)

    freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=5*free_time/16) #5pi/8
    MDFE_1 = gen_MDFE_seq(free_time=3*pi/(8*dR), rabi_freq=rabi_freq, time_steps=time_steps) #3pi/8
    freevolve_2 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=free_time/2) #pi
    freevolve_3 = FreeEvolution(laser_det=(np.full(time_steps, -detuning)), duration=13*free_time/16) #13pi/8
    pulse_1 = UpPulse(laser_det=np.full(time_steps, dR), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time) #up 2pi pulse, phase = 0

    exchange21.add_pulses([
                        freevolve_1,
                        MDFE_1,
                        swap34,
                        swap23,
                        not_gate,
                        freevolve_2,
                        not_gate,
                        swap45,
                        not_gate,
                        freevolve_2,
                        not_gate,
                        swap34,
                        antiCNOT,
                        ex10,
                        freevolve_3,
                        MDFE_1,
                        ex10,
                        antiCNOT,
                        pulse_1])
    return exchange21

def exchange21short_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):

    exchange21 = PulseSequence()

    swap23 = swap23_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    swap34 = swap34_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    swap45 = swap45_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)

    exchange21.add_pulses([
                        swap34,
                        swap23,
                        swap45,
                        swap34,
                        ])
    return exchange21
    
def RR3_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    RR3 = PulseSequence()

    ex10 = exchange10_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    ex21 = exchange21_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)

    RR3.add_pulses([ex10, ex21])

    return RR3

def RR3short2_gate(rabi_freq: float, time_steps: int, detuning: float = omega_eg/2):
    RR3 = PulseSequence()

    ex10 = exchange10_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)
    ex21 = exchange21short_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning)

    RR3.add_pulses([ex10, ex21])

    return RR3