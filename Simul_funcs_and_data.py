import numpy as np
import json
import decimal
from bisect import bisect_left

# Data used in simulation
with open('ficurves.json', 'r') as file:
    fi_curves = json.load(file)
    # F-I curves stored as a nested dictionary
    # The outer dictionary has 151 string keys representing g_ks values (in mS) from "0.0" to "1.5" in 0.01 increments
    # Each inner dictionary maps frequencies (Hz) to the current (µA) needed to trigger neuron oscillations at that frequency for the given g_ks value

with open('ficurves_keys.json', 'r') as file:
    fi_curves_keys = json.load(file)
    # A list of lists, where each inner list contains the available frequency keys (in Hz) for a specific g_ks value
    # First list corresponds to g_ks 0.0 mS and the final list corresponds to 1.5 mS, with 0.01 mS increments in each list

with open('ficurves_inh.json', 'r') as file:
    inh_currents = json.load(file)
    # Dictionary to use for applied currents of inhibitory cells
    # The dictionary has 151 string keys representing g_ks values (in mS) from "0.0" to "1.5" in 0.01 increments
    # Values are currents required to keep neurons slightly below the threshold current required to elicit spikes at the given g_ks value

# Simulation parameters
E_na = 55  # Na channel reversal potential in mV
E_k = -90  # K channel reversal potential in mV
E_l = -60  # Leak channel reversal potential in mV
E_syn_exc = 0  # Synaptic current reversal potential for excitatory cells in mV
E_syn_inh = -75  # Synaptic current reversal potential for inhibitory cells in mV

g_na = 24  # Maximum conductance of Na channel in mS
g_kd = 3  # Maximum conductance of Delayed Rectifier K channel in mS
g_l = 0.02  # Maximum conductance of Leak channel in mS

# Functions used in simulation
def create_g_ks_t(t_max, g_ks_zero_time=None, timestep = 0.1):
    """
    This function generates a g_ks_t (m-channel conductance) time series.
    For the first 1000 ms, g_ks is held constant at 1.5 mS. After that, it decreases linearly to 0 mS,
    reaching 0 at g_ks_zero_time (if specified) or at the end of the simulation (t_max).

    Inputs:
        t_max (int): Length of simulation in ms

        g_ks_zero_time (int, optional): Time in ms when g_ks should reach 0. Must be greater than 1000 ms and lower than t_max

        timestep (float, optional): Integration time step in ms (default is 0.1 ms)

    Outputs:
        g_ks_t (numpy array): Time series of g_ks values, at each simulation step
    """
    start_value = 1.5  # mS
    end_value = 0      # mS
    warm_up_time = 1000  # ms

    if g_ks_zero_time is None:
        g_ks_zero_time = t_max

    # Initialize array
    g_ks_t = start_value * np.ones(int(t_max / timestep))

    warm_up_step = int(warm_up_time / timestep)
    decline_end_step = int(g_ks_zero_time / timestep)

    # Linear decline
    g_ks_t[warm_up_step:decline_end_step] = np.linspace(start_value, end_value, decline_end_step - warm_up_step)

    # If decline ends before t_max, hold at 0 after
    if decline_end_step < len(g_ks_t):
        g_ks_t[decline_end_step:] = 0

    return g_ks_t

def m_inf(voltage):
    return 1 / (1 + np.exp((-voltage - 30) / 9.5))

def h_inf(voltage):
    return 1 / (1 + np.exp((voltage + 53) / 7.0))

def n_inf(voltage):
    return 1 / (1 + np.exp((-voltage - 30) / 10))

def z_inf(voltage):
    return 1 / (1 + np.exp((-voltage - 39) / 5))

def tau_h(voltage):
    return 0.37 + 2.78 / (1 + np.exp((voltage + 40.5) / 6))

def tau_n(voltage):
    return 0.37 + 1.85 / (1 + np.exp((voltage + 27) / 15))

tau_z = 75

def rk_slope(voltage, app_current, syn_current, h_gate, n_gate, z_gate, g_ks, timestep = 0.1):
    """
    Computes the change in state variables using the 4th-order Runge-Kutta method for one time step,
    for the neuron model described in the Materials and Methods section.

    Inputs:
        voltage (float): Membrane potential (mV)

        app_current (float): Applied current (µA)

        syn_current (float): Synaptic current (µA)

        h_gate (float): h gating variable

        n_gate (float): n gating variable

        z_gate (float): z gating variable

        g_ks (float): m-channel conductance (mS)

        timestep (float, optional): Integration time step in ms (default is 0.1 ms)

    Outputs:
        dh (float): Change in h gating variable

        dn (float): Change in n gating variable

        dz (float): Change in z gating variable

        dv (float): Change in membrane potential
    """


    k_h1 = timestep * (h_inf(voltage) - h_gate) / tau_h(voltage)
    k_n1 = timestep * (n_inf(voltage) - n_gate) / tau_n(voltage)
    k_z1 = timestep * (z_inf(voltage) - z_gate) / tau_z
    k_v1 = timestep * (-g_na * ((m_inf(voltage)) ** 3) * h_gate * (voltage - E_na) - g_kd * (n_gate ** 4) * (
                voltage - E_k) - g_ks * z_gate * (voltage - E_k) - g_l * (voltage - E_l) + app_current - syn_current)

    k_h2 = timestep * (h_inf(voltage + k_v1 / 2) - (h_gate + k_h1 / 2)) / tau_h(voltage + k_v1 / 2)
    k_n2 = timestep * (n_inf(voltage + k_v1 / 2) - (n_gate + k_n1 / 2)) / tau_n(voltage + k_v1 / 2)
    k_z2 = timestep * (z_inf(voltage + k_v1 / 2) - (z_gate + k_z1 / 2)) / tau_z
    k_v2 = timestep * (-g_na * ((m_inf(voltage + k_v1 / 2)) ** 3) * (h_gate + k_h1 / 2) * (
                voltage + k_v1 / 2 - E_na) - g_kd * ((n_gate + k_n1 / 2) ** 4) * (voltage + k_v1 / 2 - E_k) - g_ks * (
                             z_gate + k_z1 / 2) * (voltage + k_v1 / 2 - E_k) - g_l * (
                             voltage + k_v1 / 2 - E_l) + app_current - syn_current)

    k_h3 = timestep * (h_inf(voltage + k_v2 / 2) - (h_gate + k_h2 / 2)) / tau_h(voltage + k_v2 / 2)
    k_n3 = timestep * (n_inf(voltage + k_v2 / 2) - (n_gate + k_n2 / 2)) / tau_n(voltage + k_v2 / 2)
    k_z3 = timestep * (z_inf(voltage + k_v2 / 2) - (z_gate + k_z2 / 2)) / tau_z
    k_v3 = timestep * (-g_na * ((m_inf(voltage + k_v2 / 2)) ** 3) * (h_gate + k_h2 / 2) * (
                voltage + k_v2 / 2 - E_na) - g_kd * ((n_gate + k_n2 / 2) ** 4) * (voltage + k_v2 / 2 - E_k) - g_ks * (
                             z_gate + k_z2 / 2) * (voltage + k_v2 / 2 - E_k) - g_l * (
                             voltage + k_v2 / 2 - E_l) + app_current - syn_current)

    k_h4 = timestep * (h_inf(voltage + k_v3) - (h_gate + k_h3)) / tau_h(voltage + k_v3)
    k_n4 = timestep * (n_inf(voltage + k_v3) - (n_gate + k_n3)) / tau_n(voltage + k_v3)
    k_z4 = timestep * (z_inf(voltage + k_v3) - (z_gate + k_z3)) / tau_z
    k_v4 = timestep * (-g_na * ((m_inf(voltage + k_v3)) ** 3) * (h_gate + k_h3) * (voltage + k_v3 - E_na) - g_kd * (
                (n_gate + k_n3) ** 4) * (voltage + k_v3 - E_k) - g_ks * (z_gate + k_z3) * (
                             voltage + k_v3 - E_k) - g_l * (voltage + k_v3 - E_l) + app_current - syn_current)

    dh = (k_h1 + 2 * k_h2 + 2 * k_h3 + k_h4) / 6
    dn = (k_n1 + 2 * k_n2 + 2 * k_n3 + k_n4) / 6
    dz = (k_z1 + 2 * k_z2 + 2 * k_z3 + k_z4) / 6
    dv = (k_v1 + 2 * k_v2 + 2 * k_v3 + k_v4) / 6

    return dh, dn, dz, dv

def take_closest(myList, myNumber):
    """
    Returns the value in a sorted list that is closest to a given number.
    If two values are equally close, the smaller one is returned.

    Inputs:
        myList (list of floats or ints): A sorted list of numbers.

        myNumber (float or int): The target number to find the closest value to.

    Outputs:
        float or int: The value from myList closest to myNumber.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

# Look up tables for exponential functions used for synaptic current
# See equation under Network Structure in Materials and Methods
tau_r = 0.2 # ms
tau_d_e = 3 # for excitarory synapses, in ms
tau_d_i = 5.5 # for inhibitory synapses, in ms

double_exp_list1 = [0] * 5001 # Look up table for excitatory synapses
for num in range(0, 5001):
    double_exp_list1[num] = np.exp(-(num * 0.01) / tau_d_e) - np.exp(-(num * 0.01) / tau_r)

double_exp_list2 = [0] * 5001 # Look up table for inhibitory synapses
for num in range(0, 5001):
    double_exp_list2[num] = np.exp(-(num * 0.01) / tau_d_i) - np.exp(-(num * 0.01) / tau_r)



def record_spike(voltage, spike_threshold, step,
                           should_record_spike, neuron_list, timestep = 0.1):
    """
    Records neuron's spike times based on its membrane potential.

    Inputs:
        voltage (float): Membrane potential of neuron at a given time (mV)

        spike_threshold (int): Threshold voltage in mV above which a spike is recorded

        step (int): Simulation time step

        should_record_spike (Boolean): Flag indicating if neuron can currently record a spike

        neuron_list (list of floats): Neuron spikes times in ms

        timestep (float, optional): Integration time step in ms (default is 0.1 ms)

    Outputs:
        should_record_spike (Boolean): Flag indicating if neuron can currently record a spike.


    """
    if should_record_spike:
        if voltage > spike_threshold:
            neuron_list.append(timestep * step) # Record spike
            should_record_spike = False # Wait until voltage goes below spike threshold before detecting next spike
    else:
        if voltage < spike_threshold:
            should_record_spike = True # Reset should_record_spike if neuron voltage falls below spike threshold

    return should_record_spike

def simulation(EI_connectivity_strength, IE_connectivity_strength, II_connectivity_strength,
               EE_connectivity_strength, current_modulation, inh_modulation, t_max, dt = 0.1, static_g_ks=None, IE_connectivity_shift=None, g_ks_zero_time=None):
    """
    This function sets up a network with 800 excitatory and 200 inhibitory neurons,
    using the equations specified under Neuron Model in Materials and Methods.
    Numerical solution is calculated over the specified time period.
    Cholinergic modulation is introduced through g_ks linear decline. By default, g_ks is set to reach 0 mS at t_max.
    Current modulation can be enabled to keep firing frequencies of neurons approximately constant, accounting for cholinergic modulation's influence on neuronal
    excitability.

    Inputs:
        EI_connectivity_strength (float): Synaptic weight of excitatory to inhibitory connections in mS

        IE_connectivity_strength (float): Synaptic weight of inhibitory to excitatory connections in mS

        II_connectivity_strength (float): Synaptic weight of inhibitory to inhibitory connections in mS

        EE_connectivity_strength (float): Synaptic weight of excitatory to excitatory connections in mS

        current_modulation (Boolean): Flag to enable/disable current modulation for all neurons

        inh_modulation (Boolean): Flag to enable/disable inhibitory neuron g_ks modulation. If set to 0, inhibitory cells' g_ks is set to 0 mS.

        t_max (int): Length of simulation in ms

        dt (float, optional): Integration time step in ms (default is 0.1 ms)

        static g_ks (float, optional): Specify to set g_ks to any constant value in mS for all cells (will not override inh_modulation).

        IE_connectivity_shift (tuple (int, float), optional): Allows modulation of the IE synaptic weight during the simulation. The first element of the tuple specifies the time (in ms) at which the change should occur, and the second element specifies the magnitude of the change in synaptic weight (in mS) as a float.

        g_ks_zero_time (int, optional): Time in ms when g_ks should reach 0. Must be greater than 1000 ms and lower than t_max

    Outputs:
        neuron_list_exc_sorted (dict of dicts):
            Nested dictionary containing excitatory neurons sorted by firing frequency.
            - 800 string keys ("0" to "799")
            - Each key maps to a dictionary with:
                - "spike times": List of floats representing spike times in ms in chronological order.

        neuron_list_inh (dict of dicts):
            Nested dictionary containing inhibitory neurons sorted by firing frequency.
            - 200 string keys ("0" to "199")
            - Each key maps to a dictionary with:
                - "spike times": List of floats representing spike times in ms in chronological order.

        exc_currs (list of floats):
            Applied current in µA to an excitatory neuron with average firing frequency at each simulation step.

        g_ks_t (numpy array): Time series of g_ks values, at each simulation step.
    """

    # Total number of steps in simulation
    steps = int(t_max / dt)

    #Computing g_ks_t
    g_ks_t = create_g_ks_t(t_max, timestep=dt) # Sets rate of g_ks linear decline such that g_ks reaches 0 at t_max

    if g_ks_zero_time !=None:
        g_ks_t = create_g_ks_t(t_max, timestep=dt, g_ks_zero_time=g_ks_zero_time)

    if static_g_ks != None:
        g_ks_t = np.ones(steps) * static_g_ks

    # Generate Neurons
    number_of_neurons = 1000
    number_of_exc_neurons = 800  # 200 inhibitory cells

    # Initialize data related to each excitatory neuron - current, spike times, selected firing frequency
    neuron_list_exc = {
        neuron: {"current": 0, "spike times": [], "frequency": 0}
        for neuron in range(number_of_exc_neurons)
    }

    # Initialize data related to each inhibitory neuron - current, spike times, selected random value to modify current
    neuron_list_inh = {
        neuron: {"current": 0, "spike times": [], "current random seed": 0}
        for neuron in range(number_of_neurons - number_of_exc_neurons)
    }

    # Initialize connectivity matrix
    g_syn = np.zeros((number_of_neurons, number_of_neurons))

    # Create EE connections (excitatory to excitatory)
    for index in np.ndindex((number_of_exc_neurons, number_of_exc_neurons)):
        probability_of_synapse = 0.3
        g_syn[index] = np.random.choice(
            [EE_connectivity_strength, 0],
            p=[probability_of_synapse, 1 - probability_of_synapse]
        )
        if index[0] == index[1]:  # No self-synapses
            g_syn[index] = 0

    # Create IE connections (inhibitory to excitatory)
    for index in np.ndindex((number_of_exc_neurons, number_of_neurons - number_of_exc_neurons)):
        probability_of_synapse = 0.5
        g_syn[index[0], index[1] + number_of_exc_neurons] = np.random.choice(
            [IE_connectivity_strength, 0],
            p=[probability_of_synapse, 1 - probability_of_synapse]
        )

    # Create EI connections (excitatory to inhibitory)
    for index in np.ndindex((number_of_neurons - number_of_exc_neurons, number_of_exc_neurons)):
        probability_of_synapse = 0.5
        g_syn[index[0] + number_of_exc_neurons, index[1]] = np.random.choice(
            [EI_connectivity_strength, 0],
            p=[probability_of_synapse, 1 - probability_of_synapse]
        )

    # Create II Connections (inhibitory to inhibitory)
    for index in np.ndindex((number_of_neurons - number_of_exc_neurons, number_of_neurons - number_of_exc_neurons)):
        probability_of_synapse = 0.3
        g_syn[index[0] + number_of_exc_neurons, index[1] + number_of_exc_neurons] = np.random.choice(
            [II_connectivity_strength, 0],
            p=[probability_of_synapse, 1 - probability_of_synapse]
        )
        if index[0] == index[1]:  # No self-synapses
            g_syn[index[0] + number_of_exc_neurons, index[1] + number_of_exc_neurons] = 0

    if IE_connectivity_shift is not None: # Set up IE perturbation if specified
        perturb_time_ms, perturb_weight = IE_connectivity_shift
        perturb_step = int(perturb_time_ms / dt)

    # Set Applied Current
    # Excitatory neurons are selected to fire at a frequency randomly between 45 and 55Hz
    for neuron_no in range(number_of_exc_neurons):
        neuron_list_exc[neuron_no]["frequency"] = np.random.uniform(45, 55)

    # Set a single excitatory neuron (402) firing frequency to 50Hz to track its applied current over time
    neuron_list_exc[402]["frequency"] = 50

    # Inhibitory neurons have applied current set slightly below threshold current required to fire, mulitplied by a random modifier
    for neuron_no in range(number_of_neurons - number_of_exc_neurons):
        neuron_list_inh[neuron_no]["current random seed"] = np.random.uniform(0.90476, 1) #Random modifier

    # Initialize simulation parameters
    i_hyp = np.zeros(number_of_neurons)
    v = np.zeros((number_of_neurons, 5))  # Voltage
    h = np.zeros((number_of_neurons, 5))  # h-gate
    z = np.zeros((number_of_neurons, 5))  # z-gate
    n = np.zeros((number_of_neurons, 5))  # n-gate

    # Set random initial conditions for all neurons
    for neuron_no in range(number_of_neurons):
        v[neuron_no, 0] = np.random.uniform(-62, -22) #mV
        h[neuron_no, 0] = np.random.uniform(0.2, 0.8)
        z[neuron_no, 0] = np.random.uniform(0.15, 0.25)
        n[neuron_no, 0] = np.random.uniform(0.2, 0.8)

    spike_threshold = 0  # Spikes are detected when voltage crosses 0 mV
    should_record_spike = [True] * number_of_neurons  # Flag for spike detection
    g_syn_original = np.copy(g_syn)  # Store original g_syn matrix before modification

    # Decimal for exact representation to avoid floating-point errors
    zero_point_zero_one = decimal.Decimal('0.01')

    # Initialize list which will contain values of applied currents to neuron 402 at each step
    exc_currs = [0]

    # Computing numerical solution
    for i in range(1, steps): # Loop over each time step in simulation
        update_no = i % 5  # Updating modulo 5 index to remember only 5 values of v,h,z,n at a time
        rounded_g_ks = decimal.Decimal(str(round(g_ks_t[i], 2)))

        # Calculate 'i_hyp': component of synaptic current from spike times
        # (see equation under Network Structure in Materials and Methods)
        for neuron_no in range(number_of_exc_neurons):
            for spike_time in neuron_list_exc[neuron_no]["spike times"]:
                if (dt * i - spike_time) < 50 and dt * i > 100:
                    time_difference = dt * i - spike_time
                    i_hyp[neuron_no] += double_exp_list1[int(round(time_difference, 2) * 100)]

        for neuron_no in range(number_of_neurons - number_of_exc_neurons):
            for spike_time in neuron_list_inh[neuron_no]["spike times"]:
                if (dt * i - spike_time) < 50 and dt * i > 100:
                    time_difference = dt * i - spike_time
                    i_hyp[neuron_no + number_of_exc_neurons] += double_exp_list2[int(round(time_difference, 2) * 100)]

        # IE Connectivity shift applied at required time if needed
        if IE_connectivity_shift is not None and i >= perturb_step:
            ie_g_syn = g_syn[:number_of_exc_neurons, number_of_exc_neurons:number_of_neurons]
            nonzero_indices = np.nonzero(ie_g_syn)
            ie_g_syn[nonzero_indices] += perturb_weight

        # Modify g_syn connectivity matrix to include respective (voltage - E_syn) term
        # See equation under Network Structure in Materials and Methods

        # EE connections
        g_syn[:number_of_exc_neurons, :number_of_exc_neurons] *= np.reshape(
            np.repeat((v[:number_of_exc_neurons, update_no - 1] - E_syn_exc), number_of_exc_neurons),
            (number_of_exc_neurons, number_of_exc_neurons)
        )

        # IE connections
        g_syn[:number_of_exc_neurons, number_of_exc_neurons:number_of_neurons] *= np.reshape(
            np.repeat((v[:number_of_exc_neurons, update_no - 1] - E_syn_inh),
                      number_of_neurons - number_of_exc_neurons),
            (number_of_exc_neurons, number_of_neurons - number_of_exc_neurons)
        )

        # EI connections
        g_syn[number_of_exc_neurons:number_of_neurons, :number_of_exc_neurons] *= np.reshape(
            np.repeat((v[number_of_exc_neurons:number_of_neurons, update_no - 1] - E_syn_exc),
                      number_of_exc_neurons),
            (number_of_neurons - number_of_exc_neurons, number_of_exc_neurons)
        )

        # II connections
        g_syn[number_of_exc_neurons:number_of_neurons, number_of_exc_neurons:number_of_neurons] *= np.reshape(
            np.repeat((v[number_of_exc_neurons:number_of_neurons, update_no - 1] - E_syn_inh),
                      number_of_neurons - number_of_exc_neurons),
            (number_of_neurons - number_of_exc_neurons, number_of_neurons - number_of_exc_neurons)
        )


        # Synaptic current calculation
        i_syn = g_syn @ i_hyp

        # Update each excitatory neuron
        for neuron_no in range(number_of_exc_neurons):
            # Set applied current based on whether current modulation is applied
            if current_modulation: # Selects a current that preserves selected firing frequency
                neuron_list_exc[neuron_no]["current"] = fi_curves[str(rounded_g_ks)][
                    str(take_closest(fi_curves_keys[int(rounded_g_ks / zero_point_zero_one)],
                                     neuron_list_exc[neuron_no]["frequency"]))
                ]
            else:
                neuron_list_exc[neuron_no]["current"] = fi_curves[str(1.5)][
                    str(take_closest(fi_curves_keys[150],
                                     neuron_list_exc[neuron_no]["frequency"]))]


            # Store neuron 402's applied current
            if neuron_no == 402:
                exc_currs.append(neuron_list_exc[neuron_no]["current"])

            # Runge-Kutta step
            dh, dn, dz, dv = rk_slope(
                v[neuron_no, update_no - 1],
                neuron_list_exc[neuron_no]["current"],
                i_syn[neuron_no],
                h[neuron_no, update_no - 1],
                n[neuron_no, update_no - 1],
                z[neuron_no, update_no - 1],
                g_ks_t[i],
                timestep = dt
            )

            h[neuron_no, update_no] = h[neuron_no, update_no - 1] + dh
            n[neuron_no, update_no] = n[neuron_no, update_no - 1] + dn
            z[neuron_no, update_no] = z[neuron_no, update_no - 1] + dz
            v[neuron_no, update_no] = v[neuron_no, update_no - 1] + dv

            # Store spike time if spike is triggered and modify spike detection flag
            should_record_spike[neuron_no] = record_spike(v[neuron_no, update_no], spike_threshold, i, should_record_spike[neuron_no], neuron_list_exc[neuron_no]["spike times"], timestep=dt)

        # Update each inhibitory neuron
        for neuron_no in range(number_of_exc_neurons, number_of_neurons):
            inh_neuron_no = neuron_no - number_of_exc_neurons

            # Set applied current based on modulation parameters
            if current_modulation and inh_modulation:
                neuron_list_inh[inh_neuron_no]["current"] = (
                        inh_currents[str(rounded_g_ks)] *
                        neuron_list_inh[inh_neuron_no]["current random seed"]
                )
            elif not current_modulation and inh_modulation:
                neuron_list_inh[inh_neuron_no]["current"] = (
                        inh_currents[str(1.5)] *
                        neuron_list_inh[inh_neuron_no]["current random seed"]
                )
            else:
                neuron_list_inh[inh_neuron_no]["current"] = (
                        inh_currents[str(0.0)] *
                        neuron_list_inh[inh_neuron_no]["current random seed"]
                )

            # Runge-Kutta step based on whether g_ks modulation is applied
            if inh_modulation:
                dh, dn, dz, dv = rk_slope(
                    v[neuron_no, update_no - 1],
                    neuron_list_inh[inh_neuron_no]["current"],
                    i_syn[neuron_no],
                    h[neuron_no, update_no - 1],
                    n[neuron_no, update_no - 1],
                    z[neuron_no, update_no - 1],
                    g_ks_t[i],
                    timestep = dt
                )

            else:
                dh, dn, dz, dv = rk_slope(
                    v[neuron_no, update_no - 1],
                    neuron_list_inh[inh_neuron_no]["current"],
                    i_syn[neuron_no],
                    h[neuron_no, update_no - 1],
                    n[neuron_no, update_no - 1],
                    z[neuron_no, update_no - 1],
                    0,
                    timestep = dt
                )

            h[neuron_no, update_no] = h[neuron_no, update_no - 1] + dh
            n[neuron_no, update_no] = n[neuron_no, update_no - 1] + dn
            z[neuron_no, update_no] = z[neuron_no, update_no - 1] + dz
            v[neuron_no, update_no] = v[neuron_no, update_no - 1] + dv

            # Store spike time if spike is triggered and modify spike detection flag
            should_record_spike[neuron_no] = record_spike(v[neuron_no, update_no], spike_threshold, i, should_record_spike[neuron_no], neuron_list_inh[inh_neuron_no]["spike times"], timestep=dt)

        # Reset before next loop
        i_hyp[:] = 0
        np.copyto(g_syn, g_syn_original)

    # Prepare spike lists for synchrony measure computation
    exc_spike_list_for_golomb = [
        neuron_list_exc[neuron]["spike times"] for neuron in range(number_of_exc_neurons)
    ]
    inh_spike_list_for_golomb = [
        neuron_list_inh[neuron]["spike times"] for neuron in range(number_of_neurons - number_of_exc_neurons)
    ]

    # Sort excitatory neuron spike lists by firing frequency for raster plots
    neuron_list_exc_sort_order = sorted(
        neuron_list_exc.items(),
        key=lambda item: item[1]["frequency"],
        reverse=True
    )
    neuron_list_exc_sorted = {
        i: neuron_data for i, (original_index, neuron_data) in enumerate(neuron_list_exc_sort_order)
    }

    return neuron_list_exc_sorted, neuron_list_inh, exc_currs, g_ks_t

















