import numpy as np
import matplotlib.pyplot as plt
import pickle

def ackley(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - \
           np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20

# Data Generation
np.random.seed(42)
num_samples = 1000
x1 = np.random.uniform(-2, 2, num_samples)
x2 = np.random.uniform(-2, 2, num_samples)
y = ackley(x1, x2)

inputs = np.column_stack((x1, x2))
targets = y.reshape(-1, 1)

def initialize_network(input_size, hidden_sizes, output_size):
    """Initialize the weights and biases for a neural network with given layer sizes."""
    layers = []
    sizes = [input_size] + hidden_sizes + [output_size]

    for i in range(len(sizes) - 1):
        weights = np.random.uniform(-0.5, 0.5, (sizes[i + 1], sizes[i]))
        biases = np.zeros((sizes[i + 1], 1))
        layers.append((weights, biases))

    return layers

def forward_propagation(layers, input_vector):
    """Perform forward propagation through the network."""
    activations = input_vector
    cache = []

    for weights, biases in layers[:-1]:
        hidden_raw = biases + weights @ activations
        activations = 1 / (1 + np.exp(-hidden_raw))  # Sigmoid activation
        cache.append((hidden_raw, activations))

    # Output layer (linear activation)
    weights, biases = layers[-1]
    output_raw = biases + weights @ activations
    cache.append((output_raw, output_raw))  # Linear activation

    return output_raw, cache

def backpropagation(layers, cache, input_vector, target):
    """Perform backpropagation and compute gradients."""
    gradients = []

    # Output layer error
    output_raw, output = cache[-1]
    delta_output = 2 * (output - target)  # Linear layer gradient

    # Backpropagation through layers
    delta = delta_output
    for i in reversed(range(len(layers))):
        weights, biases = layers[i]
        if i > 0:
            _, activations_prev = cache[i - 1]
        else:
            activations_prev = input_vector

        # Gradients for current layer
        grad_weights = delta @ activations_prev.T
        grad_biases = delta

        gradients.insert(0, (grad_weights, grad_biases))

        # Compute delta for the previous layer if not input layer
        if i > 0:
            hidden_raw, activations = cache[i - 1]
            delta = (weights.T @ delta) * (activations * (1 - activations))

    return gradients

def update_parameters(layers, gradients, learning_rate):
    """Update network parameters using gradients."""
    for i in range(len(layers)):
        weights, biases = layers[i]
        grad_weights, grad_biases = gradients[i]
        layers[i] = (weights - learning_rate * grad_weights, biases - learning_rate * grad_biases)

def save_network(layers, filename):
    """Save the trained network to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(layers, f)

def load_network(filename):
    """Load a network from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    print("Welcome to the Neural Network Trainer")
    input_size = 2
    output_size = 1

    # User input for loading or training
    choice = input("Do you want to load a saved network? (yes/no): ").strip().lower()
    if choice == "yes":
        load_file = input("Enter filename to load the network (e.g., network.pkl): ").strip()
        try:
            layers = load_network(load_file)
            print(f"Network loaded from {load_file}")
        except FileNotFoundError:
            print(f"File {load_file} not found. Exiting.")
            return
    else:
        # User input for configuration
        hidden_layer_config = input("Enter hidden layer configuration (e.g., 20,10 for two layers): ").strip()
        hidden_sizes = [int(x) for x in hidden_layer_config.split(",")]
        epochs = int(input("Enter number of epochs: "))
        learning_rate = float(input("Enter learning rate: "))
        save_file = input("Enter filename to save the network (e.g., network.pkl): ").strip()

        # Initialize network
        layers = initialize_network(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)

        for epoch in range(epochs):
            e_loss = 0

            for i in range(num_samples):
                input_vector = inputs[i].reshape(-1, 1)
                target = targets[i]

                # Forward propagation
                output, cache = forward_propagation(layers, input_vector)

                # Error calculation
                loss = (output - target) ** 2
                e_loss += loss

                # Backpropagation
                gradients = backpropagation(layers, cache, input_vector, target)

                # Update parameters
                update_parameters(layers, gradients, learning_rate)

            # Average error on epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {e_loss[0] / num_samples}")

        # Save trained network
        save_network(layers, save_file)
        print(f"Network saved to {save_file}")

    # Testing on a grid
    grid_x1 = np.arange(-2, 2.01, 0.01)
    grid_x2 = np.arange(-2, 2.01, 0.01)
    grid_x1, grid_x2 = np.meshgrid(grid_x1, grid_x2)

    actual_values = ackley(grid_x1, grid_x2)
    approx_values = np.zeros_like(actual_values)

    for i in range(grid_x1.shape[0]):
        for j in range(grid_x1.shape[1]):
            input_vector = np.array([grid_x1[i, j], grid_x2[i, j]]).reshape(-1, 1)
            approx_values[i, j], _ = forward_propagation(layers, input_vector)

    # Visualization
    plt.figure(figsize=(10, 5))

    # Actual values
    plt.subplot(1, 2, 1)
    plt.contourf(grid_x1, grid_x2, actual_values, levels=50, cmap='viridis')
    plt.title("Rzeczywiste wartości funkcji Ackley'a")
    plt.colorbar()

    # Approx values
    plt.subplot(1, 2, 2)
    plt.contourf(grid_x1, grid_x2, approx_values, levels=50, cmap='viridis')
    plt.title("Aproksymowane wartości funkcji")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # MSE calculation
    mse = np.mean((approx_values - actual_values) ** 2)
    print(f"Błąd średniokwadratowy (MSE): {mse}")

if __name__ == "__main__":
    main()
