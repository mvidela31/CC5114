from neural_networks import Perceptron

class NANDPerceptron(Perceptron.Perceptron):
    """Object that inherits from Perceptron implementing the NAND logic gate."""
    def __init__(self):
        """Set the Weights attribute on [-2, -2] and the Bias attribute on 3."""
        super(NANDPerceptron, self).__init__(2)
        self.setWeight(0, -2)
        self.setWeight(1, -2)
        self.setBias(3)