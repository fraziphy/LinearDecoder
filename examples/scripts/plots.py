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
    ax.set_title('Spike Raster Plot for a Sample Trial')

    # Adjust y-axis limits to show all neurons
    # The 0.5 offset creates a small margin above and below the plotted neurons
    ax.set_ylim(min(neuron_ids) - 0.5, max(neuron_ids) + 0.5)

    # Display the plot
    plt.show()


def plot_results(signal, prediction_train=None, prediction_test=None):
    """
    Plot original signals and their predictions if available.

    Args:
    signal (np.array): Original signal array of shape (2, time_steps)
    prediction_train (np.array, optional): Predicted signal for training set
    prediction_test (np.array, optional): Predicted signal for test set
    """
    plt.figure(figsize=(12, 6))  # Create a new figure with specified size

    # Plot original signals
    plt.plot(signal[0], "-k", label="Auditory Stimulus")  # Plot auditory stimulus in black solid line
    plt.plot(signal[1], "--k", label="Visual Stimulus")  # Plot visual stimulus in black dashed line

    # If predictions are provided, plot them
    if prediction_train is not None:
        plt.plot(prediction_train[0,:,0], "b", alpha=0.6, label="Decoded Auditory Stimulus (Train set)")  # Plot predicted auditory stimulus for train set
        plt.plot(prediction_train[0,:,1], "r", alpha=0.6, label="Decoded Visual Stimulus (Train set)")  # Plot predicted visual stimulus for train set

        plt.plot(prediction_test[0,:,0], "c", alpha=0.6, label="Decoded Auditory Stimulus (Test set)")  # Plot predicted auditory stimulus for test set
        plt.plot(prediction_test[0,:,1], "m", alpha=0.6, label="Decoded Visual Stimulus (Test set)")  # Plot predicted visual stimulus for test set

    plt.legend()  # Add legend to the plot
    plt.xlabel("Time Steps")  # Label x-axis
    plt.ylabel("Signal Amplitude")  # Label y-axis
    if prediction_train is not None:
        plt.title("Original and Decoded Signals")  # Add title to the plot
    else:
        plt.title("Original Signals")  # Add title to the plot
    plt.show()  # Display the plot
