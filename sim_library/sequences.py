import numpy as np
from scipy.constants import pi
from sim_library.constants import dR

class Pulse:
    """
    Base class representing one of the three fundamental operations of the MSQC (up pulse, down pulse, free evolution).

    Attributes:
        laser_det (float or np.ndarray): Two-photon detuning of laser in 2pi*Hz.
        duration (float): Length of pulse/free evolution in seconds.
    """
    laser_det: float | np.ndarray
    duration: float
    def __init__(self, laser_det, duration):
        self.laser_det = laser_det
        self.duration = duration

class UpPulse(Pulse):
    """
    Represents a Raman pulse in the positive direction.

    Inherits from Pulse

    Attributes:
        phase (float or np.ndarray): Laser phase, specifically the phase difference between both Raman components.
        rabi_freq (float or np.ndarray): Rabi frequency of pulse in 2pi*Hz.
        type (str): Identifier string, fixed as 'up'.
        type_int (int): Integer identifier, fixed as 0.
    """
    phase: float | np.ndarray
    rabi_freq: float | np.ndarray
    type: str
    type_int: int
    def __init__(self, laser_det, phase, rabi_freq, duration):
        super().__init__(laser_det, duration)
        self.phase = phase
        self.rabi_freq = rabi_freq
        self.type = 'up'
        self.type_int = 0

class DownPulse(Pulse):
    """
    Represents a Raman pulse in the negative direction.

    Inherits from Pulse

    Attributes:
        phase (float or np.ndarray): Laser phase, specifically the phase difference between both Raman components.
        rabi_freq (float or np.ndarray): Rabi frequency of pulse in 2pi*Hz.
        type (str): Identifier string, fixed as 'down'.
        type_int (int): Integer identifier, fixed as 1.
    """
    phase: float | np.ndarray
    rabi_freq: float | np.ndarray
    type: str
    type_int: int
    def __init__(self, laser_det, phase, rabi_freq, duration):
        super().__init__(laser_det, duration)
        self.phase = phase
        self.rabi_freq = rabi_freq
        self.type = 'down'
        self.type_int = 1

class FreeEvolution(Pulse):
    """
    Represents a period of free evolution.

    Inherits from Pulse

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

class PulseSequence:
    """
    Represents an ordered list of Pulse objects.

    Attributes:
        pulses (List of Pulse): List containing the ordered pulses
    """

    pulses: list
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
        
def gen_resonant_pm_seq(no_pulses: int, rabi_freq: float, p_start: int = 0, dir: str = 'pos') -> PulseSequence:
    """
    Generates sequence of pi pulses of alternating direction (starting on up pulse), with no free evolution, depending on specific number of pulses. 
    Rabi frequency stays constant for each pulse and phase is set to 0.
    Laser detuning is set to be resonant for each individual pulse.

    This code is quite janky, I could make it a more readable method but it works so I probably wont change it.

    Args:
        no_pulses (int): The total number of pulses.
        rabi_freq (float or numpy.ndarray): Rabi frequency of all pulses in 2pi*Hz.
                                            
    Returns:
        PulseSequence: An object containing the generated sequence of alternating pulses.
    """
    pulse_seq = PulseSequence()

    if dir == 'pos':
        shift = 0
    else:
        shift = 1

    p = p_start - shift
    for n in range(shift, no_pulses + shift):
        if n % 2 == 0:
            pulse_seq.add_pulse(UpPulse(laser_det=(2*p + 1)*dR, phase=0, rabi_freq=rabi_freq, duration=pi/rabi_freq))
        else:
            pulse_seq.add_pulse(DownPulse(laser_det=-1*(2*p + 1)*dR, phase=0, rabi_freq=rabi_freq, duration=pi/rabi_freq))
        p += (-1)**shift
    return pulse_seq

def gen_offresonant_pm_seq(no_pulses: int, rabi_freq: float, p_start: int = 0, dir: str = 'pos') -> PulseSequence:
    """
    Generates sequence of pi pulses of alternating direction (starting on up pulse), with no free evolution, depending on specific number of pulses. 
    Rabi frequency stays constant for each pulse and phase is set to 0.
    Laser detuning is set to be resonant with only the first pulse and constant throughout.

    This code is quite janky, I could make it a more readable method but it works so I probably wont change it.

    Args:
        no_pulses (int): The total number of pulses.
        rabi_freq (float or numpy.ndarray): Rabi frequency of all pulses in 2pi*Hz.
                                            
    Returns:
        PulseSequence: An object containing the generated sequence of alternating pulses.
    """
    pulse_seq = PulseSequence()

    if dir == 'pos':
        shift = 0
    else:
        shift = 1

    p = p_start - shift

    if shift % 2 == 0:
        detuning = (2*p + 1)*dR
    else:
        detuning = -1*(2*p + 1)*dR

    for n in range(shift, no_pulses + shift):
        if n % 2 == 0:
            pulse_seq.add_pulse(UpPulse(laser_det=detuning, phase=0, rabi_freq=rabi_freq, duration=pi/rabi_freq))
        else:
            pulse_seq.add_pulse(DownPulse(laser_det=detuning, phase=0, rabi_freq=rabi_freq, duration=pi/rabi_freq))
    return pulse_seq