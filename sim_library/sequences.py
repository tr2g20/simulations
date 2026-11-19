import numpy as np
from scipy.constants import pi
from sim_library.constants import dR, omega_eg

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

class FreeEvolution(Pulse):
    """
    Represents a period of free evolution.

    Inherits from Pulse

    The laser_det attribute is an array of values, where that value is held constant
    for a certain time interval. The time interval is equal for each value in the array and is determined from
    the duration divided by the total number of values. The laser_det as a function of time
    will look like steps, since the way things are simulated the Hamiltonian needs to be constant at each discrete 
    time interval.
    If you have a profile that you want to input that has one value per each time value (instead of per time interval) 
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

class PulseSequence:
    """
    Represents an ordered sequence of Pulse objects.

    times will have a length that is 1 longer than hams. Since the discrete Hamiltonian segments are defined
    as being constant during each time interval, instead of being defined at each time step.

    Note: times and hams are unpopulated until gen_hams is run.

    Attributes:
        pulses (list[Pulse]): Ordered list of Pulse objects.
        times (np.ndarray): A 1D array of each time step (in seconds) for the whole pulse sequence.
        rabis (np.ndarray): 1D array of Rabi frequencies concatenated across all pulses.
        detunings (np.ndarray): 1D array of laser detunings concatenated across all pulses.
        phases (np.ndarray): 1D array of laser phases concatenated across all pulses.
        hams (np.ndarray): A 3D array of the Hamiltonians at each time interval for the whole pulse sequence.
        pulse_types (np.ndarray): 1D array of integer identifiers representing the pulse type for each step.
    """
    pulses: list
    times: np.ndarray
    rabis: np.ndarray
    detunings: np.ndarray
    phases: np.ndarray
    pulse_types: np.ndarray

    def __init__(self):
        self.pulses = []
        self.rabis = np.array([])
        self.detunings = np.array([])
        self.phases = np.array([])
        self.pulse_types = np.array([])

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

    def build_seq(self):
        """
        Extracts and concatenates the detunings, phases, Rabi frequencies, types, and times 
        across all constituent pulses, storing them as class attributes.
        """
        detunings_list, phases_list, rabis_list, types_list = [], [], [], []
        
        for pulse in self.pulses:
            n_steps = len(pulse.laser_det)
            
            detunings_list.append(pulse.laser_det)
            types_list.append(np.full(n_steps, pulse.type_int, dtype=np.int32))
            
            if pulse.type == 'free':
                phases_list.append(np.zeros(n_steps, dtype=np.float64))
                rabis_list.append(np.zeros(n_steps, dtype=np.float64))
            else:
                phases_list.append(pulse.phase)
                rabis_list.append(pulse.rabi_freq)
                
        self.detunings = np.concatenate(detunings_list)
        self.phases = np.concatenate(phases_list)
        self.rabis = np.concatenate(rabis_list)
        self.pulse_types = np.concatenate(types_list)
        
        times_list = [np.array([0])]
        for pulse in self.pulses:
            pulse_n_steps = len(pulse.laser_det)
            times_list.append(np.linspace(times_list[-1][-1], times_list[-1][-1] + pulse.duration, pulse_n_steps + 1)[1:])
        self.times = np.concatenate(times_list)

def Fplus_gate(phase: float, time_steps: int, detuningfree: float = omega_eg/2):

    free_time = 2*pi/(np.abs(dR-detuningfree))
    F_dur = (phase/(2*pi))*free_time

    Fplus = FreeEvolution(laser_det=np.full(time_steps, -detuningfree), duration=F_dur)

    return Fplus

def Fminus_gate(phase: float, time_steps: int, detuningfree: float = omega_eg/2):

    free_time = 2*pi/(np.abs(dR-detuningfree))
    F_dur = (phase/(2*pi))*free_time

    Fplus = FreeEvolution(laser_det=np.full(time_steps, +detuningfree), duration=F_dur)

    return Fplus

def check_duration(duration, name, phase, rabi_freq):
    if duration < 0:
        raise ValueError(
        f"\n{name} has negative duration.\n"
        f"duration = {duration:.3e} s\n"
        f"phase = {phase/pi:.3f} π\n"
        f"rabi_freq = {rabi_freq/(2*pi):.1f} Hz\n"
        )

def Gplus_gate(phase: float, rabi_freq: float, time_steps: int, detuning: float = 0, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):
    """
    Generates PulseSequence object for G_+ gate with rectangular pulses.

    Args:
        phase (float): Phase (t/tau) of the G gate (in rad).
        rabi_freq (float): The Rabi frequency in 2*pi Hz.
        time_steps (int): Number of discrete time steps per pulse or freevolution.
        detuning (float): Laser detuning of up and down pulses. Defaults to global `0`.
        correction (bool): If true, adjusts free evolution time to account for pulse origins of the two pi pulses, 
            a previous beamsplitter and subsequent recombiner. Assumes rabi frequency is fixed during sequence. Defaults to True.

    Returns:
        PulseSequence: PulseSequence object containing the MDFE sequence.
    """
    Gplus = PulseSequence()

    rabi_time = 2*pi/rabi_freq
    G_dur = phase/dR

    if bs_corr:
        bs_adj = 1/rabi_freq
    else:
        bs_adj = 0
    if pi_corr:
        pi_adj = 2*pi/(rabi_freq)
    else:
        pi_adj = 0

    # if bs_corr:
    #     if G_dur/8 - (pi_adj/4) - (bs_adj) < 0:
    #         twopi_multiples = np.abs(( G_dur/8 - (pi_adj/4) - (bs_adj) ) / (pi/(4*dR)))
    #         G_dur += 8*np.ceil(twopi_multiples)*(2*pi/dR)
    # if (not bs_corr) and (pi_corr):
    #     if G_dur/4 - (pi_adj/2) < 0:
    #         twopi_multiples = np.abs(( G_dur/4 - (pi_adj/2) ) / (pi/(2*dR)))
    #         G_dur += 8*np.ceil(twopi_multiples)*(2*pi/dR)

    # check_duration(G_dur/8 - (pi_adj/4) - (bs_adj), "Gplus outer free evolution", phase, rabi_freq)
    # check_duration(G_dur/4 - (pi_adj/2), "Gplus middle free evolution", phase, rabi_freq)

    down_pulse = DownPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/2)
    freevolve1 = FreeEvolution(laser_det=np.full(time_steps, detuning), duration=G_dur/8 - (pi_adj/4) - (bs_adj))
    freevolve2 = FreeEvolution(laser_det=np.full(time_steps, detuning), duration=G_dur/4 - (pi_adj/2))

    freevolveFD = FreeEvolution(laser_det=np.full(time_steps, detuning), duration=G_dur - (2*bs_adj))

    if not freq_dep:
        Gplus.add_pulses([freevolve1, down_pulse, freevolve2, down_pulse, freevolve1])
    else:
        Gplus.add_pulses([freevolveFD])

    return Gplus

def Gminus_gate(phase: float, rabi_freq: float, time_steps: int, detuning: float = 0, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):
    """
    Generates PulseSequence object for G_- gate with rectangular pulses.

    Args:
        phase (float): Phase (t/tau) of the G gate (in rad).
        rabi_freq (float): The Rabi frequency in 2*pi Hz.
        time_steps (int): Number of discrete time steps per pulse or freevolution.
        detuning (float): Laser detuning of up and down pulses. Defaults to global `0`.
        correction (bool): If true, adjusts free evolution time to account for pulse origins of the two pi pulses, 
                a previous beamsplitter and subsequent recombiner. Assumes rabi frequency is fixed during sequence. Defaults to True.

    Returns:
        PulseSequence: PulseSequence object containing the MDFE sequence.
    """
    Gminus = PulseSequence()

    rabi_time = 2*pi/rabi_freq
    G_dur = phase/dR

    if bs_corr:
            bs_adj = 1/rabi_freq
    else:
        bs_adj = 0
    if pi_corr:
        pi_adj = 2*pi/(rabi_freq)
    else:
        pi_adj = 0

    if not freq_dep:
        # if bs_corr:
        #     if G_dur/8 - (pi_adj/4) - (bs_adj) < 0:
        #         twopi_multiples = np.abs(( G_dur/8 - (pi_adj/4) - (bs_adj) ) / (pi/(4*dR)))
        #         G_dur += 8*np.ceil(twopi_multiples)*(2*pi/dR)
        # if (not bs_corr) and (pi_corr):
        #     if G_dur/4 - (pi_adj/2) < 0:
        #         twopi_multiples = np.abs(( G_dur/4 - (pi_adj/2) ) / (pi/(2*dR)))
        #         G_dur += 8*np.ceil(twopi_multiples)*(2*pi/dR)
    
        # check_duration(G_dur/8 - (pi_adj/4) - (bs_adj), "Gplus outer free evolution", phase, rabi_freq)
        # check_duration(G_dur/4 - (pi_adj/2), "Gplus middle free evolution", phase, rabi_freq)

        up_pulse = UpPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/2)
        freevolve1 = FreeEvolution(laser_det=np.full(time_steps, detuning), duration=G_dur/8 - (pi_adj/4) - (bs_adj))
        freevolve2 = FreeEvolution(laser_det=np.full(time_steps, detuning), duration=G_dur/4 - (pi_adj/2))
        Gminus.add_pulses([freevolve1, up_pulse, freevolve2, up_pulse, freevolve1])
    else:
        freevolveFD = FreeEvolution(laser_det=np.full(time_steps, detuning), duration=G_dur - (2*bs_adj))
        Gminus.add_pulses([freevolveFD])
    return Gminus

def Rplus_gate(F_phase: float, G_phase: float, rabi_freq: float, time_steps: int, detuning: float = 0, detuningfree: float = omega_eg/2, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):

    Rplus = PulseSequence()

    rabi_time = 2*pi/rabi_freq

    up_pulse = UpPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4)
    Fplus = Fplus_gate(phase=F_phase, time_steps=time_steps, detuningfree=detuningfree)
    Gplus = Gplus_gate(phase=G_phase, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)

    Rplus.add_pulses([up_pulse, Fplus, Gplus, up_pulse])
    return Rplus

def Rminus_gate(F_phase: float, G_phase: float, rabi_freq: float, time_steps: int, detuning: float = 0, detuningfree: float = omega_eg/2, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):

    Rminus = PulseSequence()

    rabi_time = 2*pi/rabi_freq

    down_pulse = DownPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4)
    Fminus = Fminus_gate(phase=F_phase, time_steps=time_steps, detuningfree=detuningfree)
    Gminus = Gminus_gate(phase=G_phase, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)

    Rminus.add_pulses([down_pulse, Fminus, Gminus, down_pulse])
    return Rminus

def EX10_gate(rabi_freq: float, time_steps: int, detuning: float = 0, detuningfree: float = omega_eg/2, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):

    F_phase = 5*pi/4
    G_phase = pi/4
    
    EX10 = Rminus_gate(F_phase=F_phase, G_phase=G_phase, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, detuningfree=detuningfree, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)

    return EX10

def SW23_gate(rabi_freq: float, time_steps: int, detuning: float = 0, detuningfree: float = omega_eg/2, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):

    Ramsey_F_phase = 7*pi/8
    Ramsey_G_phase = pi/8
    F_phase = 3*pi/8
    G_phase = pi/8

    SW23 = PulseSequence()

    Rplus = Rplus_gate(F_phase=Ramsey_F_phase, G_phase=Ramsey_G_phase, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, detuningfree=detuningfree, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)
    Fplus = Fplus_gate(phase=F_phase, time_steps=time_steps, detuningfree=detuningfree)
    Gplus = Gplus_gate(phase=G_phase, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, bs_corr=False, pi_corr=pi_corr, freq_dep=freq_dep)

    SW23.add_pulses([Rplus, Fplus, Gplus, Rplus])

    return SW23

def SW34_gate(rabi_freq: float, time_steps: int, detuning: float = 0, detuningfree: float = omega_eg/2, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):

    Ramsey_F_phase = 5*pi/8
    Ramsey_G_phase = pi/8
    F_phase = pi/8
    G_phase = pi/8

    SW34 = PulseSequence()

    Rminus = Rminus_gate(F_phase=Ramsey_F_phase, G_phase=Ramsey_G_phase, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, detuningfree=detuningfree, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)
    Fminus = Fminus_gate(phase=F_phase, time_steps=time_steps, detuningfree=detuningfree)
    Gminus = Gminus_gate(phase=G_phase, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, bs_corr=False, pi_corr=pi_corr, freq_dep=freq_dep)

    SW34.add_pulses([Rminus, Fminus, Gminus, Rminus])

    return SW34

def SW45_gate(rabi_freq: float, time_steps: int, detuning: float = 0, detuningfree: float = omega_eg/2, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):

    Ramsey_F_phase = 3*pi/8
    Ramsey_G_phase = pi/8
    F_phase = 15*pi/8
    G_phase = pi/8

    SW45 = PulseSequence()

    Rplus = Rplus_gate(F_phase=Ramsey_F_phase, G_phase=Ramsey_G_phase, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, detuningfree=detuningfree, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)
    Fplus = Fplus_gate(phase=F_phase, time_steps=time_steps, detuningfree=detuningfree)
    Gplus = Gplus_gate(phase=G_phase, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, bs_corr=False, pi_corr=pi_corr, freq_dep=freq_dep)

    SW45.add_pulses([Rplus, Fplus, Gplus, Rplus])

    return SW45

def EX21_gate(rabi_freq: float, time_steps: int, detuning: float = 0, detuningfree: float = omega_eg/2, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):

    EX21 = PulseSequence()

    SW23 = SW23_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, detuningfree=detuningfree, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)
    SW34 = SW34_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, detuningfree=detuningfree, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)
    SW45 = SW45_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, detuningfree=detuningfree, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)

    EX21.add_pulses([SW34, SW23, SW45, SW34])

    return EX21

def RR3_gate(rabi_freq: float, time_steps: int, detuning: float = 0, detuningfree: float = omega_eg/2, bs_corr: bool = True, pi_corr: bool = True, freq_dep: bool = False):

    RR3 = PulseSequence()

    EX10 = EX10_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, detuningfree=detuningfree, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)
    EX21 = EX21_gate(rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning, detuningfree=detuningfree, bs_corr=bs_corr, pi_corr=pi_corr, freq_dep=freq_dep)

    RR3.add_pulses([EX10, EX21])

    return RR3

# def RR3_phasecomp_500kHz(time_steps: int, detuning: float = 0, detuningfree: float = omega_eg/2):
#     """
#     RR3 sequence using adjusted MDFE durations optimised for 500kHz Rabi frequency.
#     """
#     rabi_freq = 2*pi*5e5
#     rabi_time = 2*pi/rabi_freq
#     free_time = 2*pi/(np.abs(dR-detuningfree))

#     lamda = 0.039258728343342396
#     F_bs = (0.0098281 - (lamda/4))*free_time/2
#     G_bs = (lamda/4)*pi/dR

#     F_MDFE_quartpi_down = 1.5064264264264264*free_time/2
#     G_MDFE_quartpi_down = 0.6267267267267267*pi/dR

#     F_MDFE_quartpi_up = 0.5527227227227227*free_time/2
#     G_MDFE_quartpi_up = 1.6307607607607606*pi/dR

#     F_MDFE_eighthpi_down = 0.032532532532532535*free_time/2
#     G_MDFE_eighthpi_down = 0.0*pi/dR

#     F_MDFE_eighthpi_up = 1.972172172172172*free_time/2
#     G_MDFE_eighthpi_up = 0.0*pi/dR

#     ex10 = PulseSequence()
#     pulse_1 = DownPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4) # down pi/2, phi=0
#     MDFE_1 = gen_MDFEup_seq(free_time=G_MDFE_quartpi_up-(2*G_bs), rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning) # pi/4
#     freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, +detuningfree)), duration=(5*free_time/8)-F_MDFE_quartpi_up-(2*F_bs)) # 5pi/8
#     freevolve_2 = FreeEvolution(laser_det=(np.full(time_steps, +detuningfree)), duration=free_time/2) # pi
    
#     ex10.add_pulses([pulse_1,
#                     freevolve_1,
#                     MDFE_1,
#                     pulse_1,
#                     freevolve_2])

#     swap23 = PulseSequence()
#     pulse_1 = UpPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4) # up pi/2, phi=0
#     MDFE_1 = gen_MDFEdown_seq(free_time=G_MDFE_eighthpi_down-(2*G_bs), rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning) # pi/8
#     MDFE_2 = gen_MDFEdown_seq(free_time=G_MDFE_eighthpi_down, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning) # pi/8
#     freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, -detuningfree)), duration=(7*free_time/16)+F_MDFE_eighthpi_down-(2*F_bs)) # 7pi/8
#     freevolve_2 = FreeEvolution(laser_det=(np.full(time_steps, -detuningfree)), duration=(3*free_time/16)+F_MDFE_eighthpi_down) # 3pi/8
#     swap23.add_pulses([pulse_1,
#                         freevolve_1,
#                         MDFE_1,
#                         pulse_1,
#                         freevolve_2,
#                         MDFE_2,
#                         pulse_1,
#                         freevolve_1,
#                         MDFE_1,
#                         pulse_1
#                         ])

#     swap34 = PulseSequence()
#     pulse_1 = DownPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4) # down pi/2, phi=0
#     MDFE_1 = gen_MDFEup_seq(free_time=G_MDFE_eighthpi_up-(2*G_bs), rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning) # pi/8
#     MDFE_2 = gen_MDFEup_seq(free_time=G_MDFE_eighthpi_up, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning) # pi/8
#     freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, +detuningfree)), duration=(5*free_time/16)-F_MDFE_eighthpi_up-(2*F_bs)) # 5pi/8
#     freevolve_2 = FreeEvolution(laser_det=(np.full(time_steps, +detuningfree)), duration=(free_time/16)-F_MDFE_eighthpi_up) # pi/8 
#     swap34.add_pulses([pulse_1,
#                         freevolve_1,
#                         MDFE_1,
#                         pulse_1,
#                         freevolve_2,
#                         MDFE_2,
#                         pulse_1,
#                         freevolve_1,
#                         MDFE_1,
#                         pulse_1,
#                         ])
   
#     swap45 = PulseSequence()
#     pulse_1 = UpPulse(laser_det=np.full(time_steps, detuning), phase=np.full(time_steps, 0), rabi_freq=np.full(time_steps, rabi_freq), duration=rabi_time/4) # up pi/2, phi=0
#     MDFE_1 = gen_MDFEdown_seq(free_time=G_MDFE_eighthpi_down-(2*G_bs), rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning) #pi/8
#     MDFE_2 = gen_MDFEdown_seq(free_time=G_MDFE_eighthpi_down, rabi_freq=rabi_freq, time_steps=time_steps, detuning=detuning) #pi/8
#     freevolve_1 = FreeEvolution(laser_det=(np.full(time_steps, -detuningfree)), duration=(3*free_time/16)+F_MDFE_eighthpi_down-(2*F_bs)) #3pi/8
#     freevolve_2 = FreeEvolution(laser_det=(np.full(time_steps, -detuningfree)), duration=(15*free_time/16)+F_MDFE_eighthpi_down) #15pi/8
#     swap45.add_pulses([pulse_1,
#                         freevolve_1,
#                         MDFE_1,
#                         pulse_1,
#                         freevolve_2,
#                         MDFE_2,
#                         pulse_1,
#                         freevolve_1,
#                         MDFE_1,
#                         pulse_1,
#                         ])

#     ex21 = PulseSequence()
#     ex21.add_pulses([
#                         swap34,
#                         swap23,
#                         swap45,
#                         swap34,
#                         ])

#     RR3 = PulseSequence()
#     RR3.add_pulses([ex10, ex21])

#     return RR3
