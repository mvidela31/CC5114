import math
from neural_networks import Perceptron

class Neuron(Perceptron.Perceptron):
    """Object that inherits from Perceptron. Basic unit of a neural network able to keep the delta and output values
       and modifies all his parameters (including the inherited attributes from Perceptron object) according to specific
       inputs."""
    def __init__(self, numberOfWeights, output = None, delta = None):
        """
        :param numberOfWeights: Size of the weights random float array attribute.
        :param output: Output value.
        :param delta: Delta value.
        """
        super(Neuron, self).__init__(numberOfWeights)
        self.output = output
        self.delta = delta

    def getOutput(self):
        """Return the output attribute of the current neuron."""
        return self.output

    def getDelta(self):
        """Return the delta attribute of the current neuron."""
        return self.delta

    def transferDerivative(self, output):
        """Calculates the derivative approximation of an output value."""
        return output * (1.0 - output)

    def updateOutput(self, input):
        """Update the output attribute of the current neuron using the sigmoid function."""
        self.output = 1.0 / (1.0 + math.exp(-self.dotProduct(input) - self.getBias()))

    def updateDelta(self, error):
        """Update the output attribute of the current neuron according to an error value using the transfer derivative
           method."""
        self.delta = error * self.transferDerivative(self.getOutput())

    def updateWeights(self, input, learningRate):
        """Update the weights array attribute of the current neuron according to an input and a learning rate."""
        for i in range(len(self.getWeights())):
            self.setWeight(i, self.getWeights()[i] + learningRate * self.getDelta() * input[i])

    def updateBias(self, learningRate):
        """Update the bias attribute of the current neuron according to its delta attribute and a learning rate
           value."""
        self.setBias(self.getBias() + learningRate * self.getDelta())