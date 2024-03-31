import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

class RBM:
    def __init__(self, input_dim, hidden_dim):
        self.weights = np.random.randn(input_dim, hidden_dim) * 0.1
        self.visible_bias = np.zeros(input_dim)
        self.hidden_bias = np.zeros(hidden_dim)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sample_h_given_v(self, v):
        p_h_given_v = self.sigmoid(np.dot(v, self.weights) + self.hidden_bias)
        return np.random.binomial(1, p_h_given_v)
    
    def sample_v_given_h(self, h):
        p_v_given_h = self.sigmoid(np.dot(h, self.weights.T) + self.visible_bias)
        return np.random.binomial(1, p_v_given_h)
    
    def contrastive_divergence(self, input_data, k=1, learning_rate=0.1):
        for _ in range(k):
            hidden_prob = self.sigmoid(np.dot(input_data, self.weights) + self.hidden_bias)
            hidden_state = np.random.binomial(1, hidden_prob)
            visible_recon = self.sigmoid(np.dot(hidden_state, self.weights.T) + self.visible_bias)
            visible_recon_prob = self.sigmoid(np.dot(visible_recon, self.weights) + self.hidden_bias)
            hidden_recon = np.random.binomial(1, visible_recon_prob)
            
            self.weights += learning_rate * (np.dot(input_data.T, hidden_prob) - np.dot(visible_recon.T, hidden_recon))
            self.visible_bias += learning_rate * np.mean(input_data - visible_recon, axis=0)
            self.hidden_bias += learning_rate * np.mean(hidden_prob - hidden_recon, axis=0)

# Example usage
X_train = np.random.rand(100, 784)  # Example training data (100 samples of 28x28 flattened images)
y_train = np.random.randint(0, 2, size=(100,))  # Example training labels (binary classification)

input_dim = X_train.shape[1]
hidden_dim = 128

# Create and train RBM
rbm = RBM(input_dim, hidden_dim)
rbm.contrastive_divergence(X_train)

# Create and compile the classifier (fine-tuning)
classifier = Sequential([
    Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
    Dense(1, activation='sigmoid')
])
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the classifier
classifier.fit(X_train, y_train, epochs=10, batch_size=32)
