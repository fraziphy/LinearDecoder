# test_dependencies.py

import unittest
from src.linear_decoder.linear_decoder import LinearDecoder

class TestLinearDecoder(unittest.TestCase):
    def setUp(self):
        # Set up any necessary test data or objects
        self.decoder = LinearDecoder(n_neurons=10, duration=1000, dt=0.1, tau=10, lambda_reg=1e-3)

    def test_initialization(self):
        self.assertIsInstance(self.decoder, LinearDecoder)
        self.assertEqual(self.decoder.n_neurons, 10)
        # Add more assertions to check initialization

    def test_preprocess_data(self):
        # Create dummy spike data
        spikes_trials_all = [[(10, 1), (20, 2), (30, 3)] for _ in range(5)]  # 5 trials
        filtered_spikes = self.decoder.preprocess_data(spikes_trials_all)
        self.assertEqual(filtered_spikes.shape, (5, 10000, 10))  # Adjust shape based on your implementation

    def test_fit_and_predict(self):
        # Create dummy data
        import numpy as np
        filtered_spikes = np.random.rand(5, 10000, 10)
        signal = np.random.rand(1, 10000)

        self.decoder.fit(filtered_spikes[:3], signal)  # Use first 3 trials for training
        prediction = self.decoder.predict(filtered_spikes[3:])  # Use last 2 trials for testing

        self.assertEqual(prediction.shape, (2, 10000))  # Adjust shape based on your implementation

    # Add more test methods as needed

if __name__ == '__main__':
    unittest.main()
