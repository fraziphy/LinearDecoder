# funcs.py

import numpy as np
from sklearn.model_selection import KFold

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


class LinearDecoder:
    _usage_guide = """
    LinearDecoder Usage Guide:

    1. Initialize the decoder:
       decoder = funcs.LinearDecoder(dt, tau, lambda_reg, rng)

       Parameters:
       - dt: Recording resolution (time step) in milliseconds
       - tau: Time constant for the exponential kernel in milliseconds
       - lambda_reg: Regularization strength to prevent overfitting
       - rng: Random number generator (e.g., np.random.default_rng(seed))

    2. Preprocess data:
       filtered_spikes = decoder.preprocess_data(spikes_trials_all, n_neurons, duration)

       Parameters:
       - n_neurons: Number of neurons in the recording
       - duration: Total duration of the recording in milliseconds

       Note: spikes_trials_all should be a list of trials, where each trial is a list of tuples (spike_time, neuron_id).
       The resulting filtered_spikes will have shape (n_trials, n_steps, n_neurons).

    3. Fit the decoder and make predictions:
       decoder.fit(filtered_spikes[training_trial_indices], signal)
       prediction = decoder.predict(filtered_spikes[test_trial_indices])
       RMSE = decoder.compute_rmse(prediction, signal)

    4. Perform stratified cross-validation:
       train_errors, test_errors, all_weights = decoder.stratified_cv(filtered_spikes, signal, n_splits=5)

       After performing stratified cross-validation, you can access:
       - decoder.example_predicted_train: An example of a predicted signal for a training trial
       - decoder.example_predicted_test: An example of a predicted signal for a test trial
       - decoder.signal: The original signal used for decoding

    Important:
    - Ensure the signal is a 2D array with dimensions (n_signals, n_time_steps).
    - Make sure the signal has the same temporal resolution as your recording (determined by dt).
    - The duration and dt parameters should match the temporal properties of your spike data and signal.
    """

    @property
    def help(self):
        print(self._usage_guide)

    def __init__(self, dt=0.1, tau=10, lambda_reg=1e-3, rng=np.random.default_rng(2)):
        """
        Initialize the LinearDecoder.

        Args:
            tau (float): Time constant for exponential kernel.
            lambda_reg (float): Regularization strength.
            rng (numpy.random.Generator): Random number generator.
        """
        self.dt = dt
        self.tau = tau
        self.lambda_reg = lambda_reg
        self.random_state = rng if isinstance(rng, int) else None
        self.w = None
        # Create exponential kernel
        self.kernel = np.exp(-np.arange(0, 5 * tau, dt) / tau)

    def preprocess_data(self, spikes_trials_all, n_neurons, duration):
        """
        Preprocess spike data: convert to matrices and apply exponential kernel.

        Args:
            spikes_trials_all (list): List of spike time tuples for each trial.
            n_neurons (int): Number of neurons.
            duration (float): Duration of the recording in ms.

        Returns:
            numpy.ndarray: Filtered spike matrices.
        """
        n_steps = int(duration / self.dt)
        spike_matrices = np.array([spikes_to_matrix(trial_spikes, n_steps, n_neurons, self.dt)
                                    for trial_spikes in spikes_trials_all])
        return filter_spikes_exp_kernel(spike_matrices, self.kernel)

    def _ensure_2d_signal(self, signal):
        """
        Ensure the signal is a 2D array.

        Args:
            signal (numpy.ndarray): Input signal.

        Returns:
            numpy.ndarray: 2D signal array.
        """
        if signal.ndim == 1:
            return signal.reshape(1, -1)
        elif signal.ndim == 2:
            return signal
        else:
            raise ValueError("Signal must be either 1D or 2D array.")

    def fit(self, filtered_spikes, signal):
        """
        Fit the linear decoder to the data.

        Args:
            filtered_spikes (numpy.ndarray): Filtered spike data.
            signal (numpy.ndarray): Signal to decode.
        """
        signal = self._ensure_2d_signal(signal)

        if filtered_spikes.ndim != 3:
            raise ValueError(f"Expected filtered_spikes to be 3D, but got shape {filtered_spikes.shape}")

        n_trials, n_steps, n_neurons = filtered_spikes.shape
        X = filtered_spikes.reshape(n_trials * n_steps, n_neurons)
        y = np.tile(signal.T, (n_trials, 1))

        # Solve the regularized least squares problem
        # (X^T X + λI) w = X^T y
        # where X is X_train, y is tiled_signal, and λ is the regularization strength
        I = np.eye(n_neurons)
        reg_term = X.T @ X + self.lambda_reg * I
        self.w = np.linalg.solve(reg_term, X.T @ y)

    def predict(self, filtered_spikes):
        """
        Make predictions using the trained decoder.

        Args:
            filtered_spikes (numpy.ndarray): Filtered spike data.

        Returns:
            numpy.ndarray: Predicted signal.
        """
        return filtered_spikes.dot(self.w)

    def compute_rmse(self, prediction, signal):
        """
        Compute Root Mean Square Error between prediction and actual signal.

        Args:
            prediction (numpy.ndarray): Predicted signal.
            signal (numpy.ndarray): Actual signal.

        Returns:
            numpy.ndarray: RMSE for each signal dimension.
        """
        return np.sqrt(((prediction - signal.T)**2).mean(axis=0))

    def stratified_cv(self, filtered_spikes, signal, n_splits=5):
        """
        Perform stratified cross-validation.

        Args:
            filtered_spikes (numpy.ndarray): Filtered spike data of shape (n_trials, n_steps, n_neurons).
            signal (numpy.ndarray): Signal to decode of shape (n_signals, n_steps).
            n_splits (int): Number of splits for cross-validation.

        Returns:
            tuple:
                - train_errors (numpy.ndarray): RMSE for training data for each fold and signal dimension.
                - test_errors (numpy.ndarray): RMSE for test data for each fold and signal dimension.
                - all_weights (list): List of weight matrices, one for each fold.
        """
        # Ensure signal is 2D
        signal = self._ensure_2d_signal(signal)

        # Initialize K-Fold cross-validator
        n_trials = filtered_spikes.shape[0]
        if n_splits == n_trials:  # LOOCV case
            kf = KFold(n_splits=n_trials, shuffle=True, random_state=self.random_state)
            train_size = n_trials - 1
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            train_size = n_trials - n_trials // n_splits

        # Initialize lists to store results
        train_errors = []
        test_errors = []
        all_weights = []  # List to store weights from each fold


        # Perform cross-validation
        for train_idx, test_idx in kf.split(filtered_spikes):
            # Prepare training data
            X_train = filtered_spikes[train_idx]

            # Fit the model
            self.fit(X_train, signal)

            # Store weights for this fold
            all_weights.append(self.w.copy())

            # Make predictions
            train_prediction = self.predict(filtered_spikes[train_idx])
            test_prediction = self.predict(filtered_spikes[test_idx])

            # Store example predictions (first fold only)
            if not hasattr(self, 'example_predicted_train'):
                self.example_predicted_train = train_prediction[0]
                self.example_predicted_test = test_prediction[0]

            # Compute and store errors
            train_errors.append(self.compute_rmse(train_prediction, signal))
            test_errors.append(self.compute_rmse(test_prediction, signal))

        # Reshape errors to (n_folds, n_signals)
        n_signals = signal.shape[0]
        train_errors = np.array(train_errors).reshape(-1, n_signals)
        test_errors = np.array(test_errors).reshape(-1, n_signals)

        return train_errors, test_errors, all_weights




# You can add more functions as needed

if __name__ == "__main__":
    print("This module contains utility functions for linear decoding.")
