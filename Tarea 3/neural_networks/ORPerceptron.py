from neural_networks import Perceptron

class ORPerceptron(Perceptron.Perceptron):
    """Object that inherits from Perceptron implementing the OR logic gate."""
    def __init__(self):
        """Set the Weights attribute on[1, 1] and the Bias attribute on 0."""
        super(ORPerceptron, self).__init__(2)
        self.setWeight(0, 1)
        self.setWeight(1, 1)
        self.setBias(0)