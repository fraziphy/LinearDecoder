# generate_data.py

import numpy as np

def dummy_spiketimes(n_neurons, duration, dt, avg_rate):
    """
    Generate dummy spike times for a population of neurons.

    Parameters:
    n_neurons (int): Number of neurons in the population.
    duration (float): Total duration of the simulation in milliseconds.
    dt (float): Time resolution in milliseconds.
    avg_rate (float): Average firing rate of neurons in Hz.

    Returns:
    list: A sorted list of tuples (spike_time, neuron_id).
    """

    # Calculate expected number of spikes per neuron
    spikes_per_neuron = int(duration * avg_rate / 1000)

    # Initialize list to store all spikes
    spike_list = []

    # Generate spikes for each neuron
    for neuron_id in range(n_neurons):
        # Generate random spike times for this neuron
        # Uniformly distributed over the duration
        neuron_spikes = np.random.rand(spikes_per_neuron) * duration

        # Round spike times to the nearest multiple of dt
        # This ensures spikes align with the time resolution
        neuron_spikes = np.round(neuron_spikes / dt) * dt

        # Add spikes to the list as tuples (spike_time, neuron_id)
        spike_list.extend([(spike_time, neuron_id) for spike_time in neuron_spikes])

    # Sort spikes by time to simulate chronological order
    spike_list.sort(key=lambda x: x[0])

    return spike_list
