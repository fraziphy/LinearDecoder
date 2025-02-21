# plots.py

import matplotlib.pyplot as plt

def plot_raster(spike_1):
    """
    Create a raster plot of neuronal spike times.

    Parameters:
    spike_1 (list): A list of tuples, where each tuple contains (spike_time, neuron_id).

    This function creates a scatter plot where:
    - Each vertical line represents a spike
    - The x-axis represents time
    - The y-axis represents individual neurons
    """

    # Unpack the spike_1 list into separate lists of spike times and neuron IDs
    spike_times, neuron_ids = zip(*spike_1)

    # Create a new figure and axis object
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the spikes as vertical lines
    # 'marker="|"' creates vertical lines, 's=10' sets the size, 'c="black"' sets the color
    ax.scatter(spike_times, neuron_ids, marker='|', s=10, c='black')

    # Label the x-axis
    ax.set_xlabel('Time (ms)')

    # Label the y-axis
    ax.set_ylabel('Neuron ID')

    # Set the title of the plot
    ax.set_title('Spike Raster Plot')

    # Adjust y-axis limits to show all neurons
    # The 0.5 offset creates a small margin above and below the plotted neurons
    ax.set_ylim(min(neuron_ids) - 0.5, max(neuron_ids) + 0.5)

    # Display the plot
    plt.show()
