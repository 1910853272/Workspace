"""ANN-SNN conversion  for the FC layer 20x4 """

import numpy as np
import pandas as pd

class SingleLayerPerceptron:
    def __init__(self, input_size, output_size):
        """Initialize weights and biases with small random values."""
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.random.randn(output_size) * 0.1

def memristor_quantize(values, bits=4):
    """Quantize the values using a linear memristor model with specified bit depth."""
    levels = 2 ** bits
    max_val = np.max(values)
    min_val = np.min(values)
    step = (max_val - min_val) / (levels - 1)
    quantized = np.round((values - min_val) / step) * step + min_val
    return quantized

def ann_to_snn(weights, biases, v_th=1.0):
    """Scale and quantize ANN weights for use in an SNN."""
    scale_factor = np.max(np.abs(weights)) / v_th
    weights_scaled = weights / scale_factor
    biases_scaled = biases / scale_factor
    weights_quantized = memristor_quantize(weights_scaled, 4)
    biases_quantized = memristor_quantize(biases_scaled, 4)
    return weights_quantized, biases_quantized

def snn_inference(weights, biases, input_frequencies, time_steps=100, dt=1.0):
    """Simulate the SNN inference given the input frequencies."""
    input_size = weights.shape[1]
    output_size = weights.shape[0]
    spike_response = np.zeros(output_size)
    for _ in range(time_steps):
        inputs = np.random.rand(input_size) < (input_frequencies * dt)
        membrane_potentials = np.dot(weights, inputs) + biases
        outputs = membrane_potentials >= 1.0
        spike_response += outputs
        biases -= outputs * 1.0
    return spike_response

def compute_accuracy(predictions, targets):
    """Calculate the accuracy of predictions against targets."""
    correct = np.sum(predictions == targets)
    return correct / len(predictions)

def load_and_process_data(filepath):
    """Load and process CSV data to calculate input frequencies based on voltage threshold."""
    df = pd.read_csv(filepath)
    threshold = 2.0  # Voltage threshold in volts
    input_frequencies = (df > threshold).sum(axis=0)  # Count each column's values above the threshold
    return input_frequencies

# Load and process data
filepath = '/opticneuronspikes/Sensoryspikes.csv'
input_frequencies = load_and_process_data(filepath)

# Define the network and test data
input_size = 20
output_size = 4
perceptron = SingleLayerPerceptron(input_size, output_size)
snn_weights, snn_biases = ann_to_snn(perceptron.weights, perceptron.biases)

# Generate ideal outputs (for testing)
ideal_outputs = (input_frequencies > 50).astype(int)[:output_size]

# Perform SNN inference
output_spikes = snn_inference(snn_weights, snn_biases, input_frequencies.values)
time_steps = 20
# Calculate inference accuracy
accuracy = compute_accuracy(output_spikes > (time_steps // 2), ideal_outputs)
print("Output spikes:", output_spikes)
print("Ideal outputs:", ideal_outputs)
print("Inference Accuracy: {:.2f}%".format(accuracy * 100))
