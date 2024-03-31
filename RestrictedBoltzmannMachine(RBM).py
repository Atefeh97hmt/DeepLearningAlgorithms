import numpy as np

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.random.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sample_h_given_v(self, visible_data):
        activation = np.dot(visible_data, self.weights) + self.hidden_bias
        p_hidden_given_visible = self.sigmoid(activation)
        return p_hidden_given_visible, np.random.binomial(1, p_hidden_given_visible)
    
    def sample_v_given_h(self, hidden_data):
        activation = np.dot(hidden_data, self.weights.T) + self.visible_bias
        p_visible_given_hidden = self.sigmoid(activation)
        return p_visible_given_hidden, np.random.binomial(1, p_visible_given_hidden)
    
    def contrastive_divergence(self, input_data, learning_rate=0.1, k=1):
        batch_size = input_data.shape[0]
        
        # Positive phase
        positive_hidden_prob, positive_hidden_state = self.sample_h_given_v(input_data)
        
        # Negative phase
        hidden_state = positive_hidden_state
        for _ in range(k):
            negative_visible_prob, negative_visible_state = self.sample_v_given_h(hidden_state)
            negative_hidden_prob, negative_hidden_state = self.sample_h_given_v(negative_visible_state)
            hidden_state = negative_hidden_state
        
        # Update parameters
        positive_associations = np.dot(input_data.T, positive_hidden_prob)
        negative_associations = np.dot(negative_visible_state.T, negative_hidden_prob)
        
        self.weights += learning_rate * ((positive_associations - negative_associations) / batch_size)
        self.visible_bias += learning_rate * np.mean(input_data - negative_visible_state, axis=0)
        self.hidden_bias += learning_rate * np.mean(positive_hidden_prob - negative_hidden_prob, axis=0)


