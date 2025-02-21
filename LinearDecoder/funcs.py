# funcs.py

from . import config
import numpy as np

def spikes_to_matrix(spike_list, n_steps, N, step_size):
    """
    Convert spike data into a spike matrix.

    Args:
        spike_list (list): List of spikes [(time, neuron_id), ...].
        n_steps (int): Number of time steps.
        N (int): Number of neurons.
        step_size (float): Time step size in ms.

    Returns:
        numpy.ndarray: Spike matrix of shape (n_steps, N).
    """
    spike_matrix = np.zeros((n_steps, N))
    for spike_time, neuron_id in spike_list:
        time_bin = int(spike_time / step_size)
        if 0 <= time_bin < n_steps and 0 <= neuron_id < N:
            spike_matrix[time_bin, neuron_id] += 1
    return spike_matrix


def fft_convolution_with_padding(signal, kernel):
    """
    Perform linear convolution using FFT with proper zero-padding.

    Args:
        signal (numpy.ndarray): Input signal.
        kernel (numpy.ndarray): Impulse response or kernel.

    Returns:
        numpy.ndarray: Linearly convolved signal.
    """
    padded_length = len(signal) + len(kernel) - 1
    padded_signal = np.pad(signal, (0, padded_length - len(signal)))
    padded_kernel = np.pad(kernel, (0, padded_length - len(kernel)))

    convolved = np.fft.ifft(np.fft.fft(padded_signal) * np.fft.fft(padded_kernel))

    return np.real(convolved[:len(signal)])


def filter_spikes_exp_kernel(spike_matrices, kernel):
    """
    Filter spike matrices using an exponential kernel through convolution.

    Args:
        spike_matrices (numpy.ndarray): Array of spike matrices with shape (n_trials, n_steps, n_neurons).
        kernel (numpy.ndarray): 1D array representing the exponential kernel.

    Returns:
        numpy.ndarray: Filtered spike matrices with the same shape as the input.
    """
    # Extract dimensions of the input spike matrices
    n_trials, n_steps, n_neurons = spike_matrices.shape

    # Apply convolution to each neuron in each trial
    filtered_spikes = np.array([
        np.array([fft_convolution_with_padding(spike_matrices[i][:, j], kernel) for j in range(n_neurons)]).T
        for i in range(n_trials)
    ])

    # The resulting filtered_spikes has the same shape as the input spike_matrices
    return filtered_spikes


def linear_decoder_training_trials(rng, filtered_spikes_training_trials, signal):
    """
    Perform linear decoding on training trials.

    Args:
        rng (numpy.random.Generator): Random number generator.
        filtered_spikes_training_trials (numpy.ndarray): Filtered spike data of shape (n_trials, n_steps, n_neurons).
        signal (numpy.ndarray): Signal to decode of shape (n_signals, n_steps).

    Returns:
        numpy.ndarray: Weights for decoding of shape (n_neurons, n_signals).
    """
    n_trials, n_steps, n_neurons = filtered_spikes_training_trials.shape
    n_signals, n_steps_signal = signal.shape

    assert n_steps == n_steps_signal, "Signal and spikes must have the same number of time steps"

    # Prepare all trials for training
    train_trials_ids = np.arange(n_trials)

    # Shuffle the training trial order
    rng.shuffle(train_trials_ids)

    # Prepare training datasets
    X_train = filtered_spikes_training_trials[train_trials_ids].reshape(-1, n_neurons)
    tiled_signal = np.tile(signal.T, (n_trials, 1))

    # Create regularization term
    I = np.eye(n_neurons)  # Identity matrix for regularization
    reg_term = X_train.T @ X_train + config.LAMBDA_REG * I

    # Solve the regularized least squares problem
    # (X^T X + λI) w = X^T y
    # where X is X_train, y is tiled_signal, and λ is the regularization strength
    w = np.linalg.solve(reg_term, X_train.T @ tiled_signal)
    return w


# You can add more functions as needed

if __name__ == "__main__":
    print("This module contains utility functions for linear decoding.")
