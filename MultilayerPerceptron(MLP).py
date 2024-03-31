import numpy as np

class MLP:
    def __init__(self, layers, learning_rate=0.01, n_iterations=1000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        for i in range(1, len(layers)):
            weight_matrix = np.random.randn(layers[i-1], layers[i])
            self.weights.append(weight_matrix)
            bias_vector = np.zeros((1, layers[i]))
            self.biases.append(bias_vector)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        activations = [X]

        # Calculate activations for each layer
        for i in range(len(self.layers) - 1):
            weighted_sum = np.dot(activations[i], self.weights[i]) + self.biases[i]
            activation = self.sigmoid(weighted_sum)
            activations.append(activation)

        return activations

    def backward_propagation(self, X, y, activations):
        deltas = [None] * (len(self.layers) - 1)

        # Calculate error and deltas for output layer
        error = y - activations[-1]
        delta = error * self.sigmoid_derivative(activations[-1])
        deltas[-1] = delta

        # Backpropagate error through hidden layers
        for i in reversed(range(len(deltas) - 1)):
            error = np.dot(deltas[i+1], self.weights[i+1].T)
            delta = error * self.sigmoid_derivative(activations[i+1])
            deltas[i] = delta

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] += self.learning_rate * np.sum(deltas[i], axis=0)

    def fit(self, X, y):
        for _ in range(self.n_iterations):
            activations = self.forward_propagation(X)
            self.backward_propagation(X, y, activations)

    def predict(self, X):
        activations = self.forward_propagation(X)
        return np.round(activations[-1])

# Example
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

mlp = MLP(layers=[2, 2, 1], learning_rate=0.1, n_iterations=10000)
mlp.fit(X_train, y_train)

X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = mlp.predict(X_test)
print("Predictions:", predictions)
