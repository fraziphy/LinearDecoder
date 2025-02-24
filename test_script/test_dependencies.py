# test_dependencies.py

import unittest
import numpy as np
from linear_decoder.linear_decoder import LinearDecoder, spikes_to_matrix, filter_spikes_exp_kernel

class TestLinearDecoder(unittest.TestCase):
    def setUp(self):
        self.decoder = LinearDecoder(dt=0.1, tau=10, lambda_reg=1e-3, rng=np.random.default_rng(2))
        self.n_neurons = 10
        self.duration = 1000  # ms
        self.n_trials = 5
        self.n_steps = int(self.duration / self.decoder.dt)

    def generate_dummy_data(self):
        spikes_trials_all = [
            [(np.random.rand() * self.duration, np.random.randint(0, self.n_neurons))
             for _ in range(50)]  # 50 spikes per trial
            for _ in range(self.n_trials)
        ]
        signal = np.random.rand(1, self.n_steps)
        return spikes_trials_all, signal

    def test_initialization(self):
        self.assertIsInstance(self.decoder, LinearDecoder)
        self.assertEqual(self.decoder.dt, 0.1)
        self.assertEqual(self.decoder.tau, 10)
        self.assertEqual(self.decoder.lambda_reg, 1e-3)

    def test_preprocess_data(self):
        spikes_trials_all, _ = self.generate_dummy_data()
        filtered_spikes = self.decoder.preprocess_data(spikes_trials_all, self.n_neurons, self.duration)
        self.assertEqual(filtered_spikes.shape, (self.n_trials, self.n_steps, self.n_neurons))

    def test_fit_and_predict(self):
        spikes_trials_all, signal = self.generate_dummy_data()
        filtered_spikes = self.decoder.preprocess_data(spikes_trials_all, self.n_neurons, self.duration)

        self.decoder.fit(filtered_spikes[:3], signal)  # Use first 3 trials for training
        prediction = self.decoder.predict(filtered_spikes[3:])  # Use last 2 trials for testing

        self.assertEqual(prediction.shape, (2, 10000, 1))

    def test_compute_rmse(self):
        _, signal = self.generate_dummy_data()
        prediction = np.random.rand(self.n_trials, self.n_steps, 1)  # Shape: (n_trials, n_steps, n_signals)
        rmse = self.decoder.compute_rmse(prediction, signal)
        self.assertIsInstance(rmse, np.ndarray)
        self.assertEqual(rmse.shape, (signal.shape[0],))  # Check if the shape matches the number of signals
        self.assertTrue(np.all(rmse >= 0))  # Ensure all RMSE values are non-negative

    def test_stratified_cv(self):
        spikes_trials_all, signal = self.generate_dummy_data()
        filtered_spikes = self.decoder.preprocess_data(spikes_trials_all, self.n_neurons, self.duration)

        train_errors, test_errors, all_weights = self.decoder.stratified_cv(filtered_spikes, signal, n_splits=5)

        self.assertEqual(train_errors.shape, (5, 1))  # 5 folds, 1 signal dimension
        self.assertEqual(test_errors.shape, (5, 1))   # 5 folds, 1 signal dimension
        self.assertEqual(len(all_weights), 5)  # 5 sets of weights

    def test_spikes_to_matrix(self):
        spike_list = [(10, 0), (20, 1), (30, 2)]
        n_steps = 100
        N = 3
        step_size = 1
        spike_matrix = spikes_to_matrix(spike_list, n_steps, N, step_size)
        self.assertEqual(spike_matrix.shape, (n_steps, N))
        self.assertEqual(spike_matrix[10, 0], 1)
        self.assertEqual(spike_matrix[20, 1], 1)
        self.assertEqual(spike_matrix[30, 2], 1)

    def test_filter_spikes_exp_kernel(self):
        spike_matrices = np.random.rand(self.n_trials, self.n_steps, self.n_neurons)
        kernel = np.exp(-np.arange(0, 50, 0.1) / 10)
        filtered_spikes = filter_spikes_exp_kernel(spike_matrices, kernel)
        self.assertEqual(filtered_spikes.shape, spike_matrices.shape)

if __name__ == '__main__':
    unittest.main()
