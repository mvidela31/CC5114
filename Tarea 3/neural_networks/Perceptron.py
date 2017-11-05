import random

class Perceptron(object):
    """Object able to make binary predictions according to an input and it own attributes values (Weights and Bias).
       Also, this object has the capacity to learn, modifying its attributes according a comparison between a desired
       output and the output obtained by this object."""
    def __init__(self, numberOfWeights):
        """
        Initializes a newly created Perceptron object with a 'weights' array of random floats attribute and a 'bias'
        random float attribute.
        :param numberOfWeights: Size of the weights random float array attribute.
        """
        weights = []
        for i in range(numberOfWeights):
            weights.append(random.uniform(-5, 5))
        self.weights = weights
        self.bias = random.uniform(-5, 5)

    def getWeights(self):
        """Return the weights attribute float array."""
        return self.weights

    def getBias(self):
        """Return the bias attribute float."""
        return self.bias

    def setWeight(self, position, weight):
        """Modifies a value of the weights float array  according to a new weight value and the position of the weight
        that will be modified."""
        self.weights[position] = weight

    def setAllWeights(self, weightsArray):
        """Modifies the entire weights attribute array to a new weights array."""
        self.weights = weightsArray

    def setBias(self, bias):
        """Modifies the bias attribute according to a new bias float value."""
        self.bias = bias

    def dotProduct(self, input):
        """Calculates the dot product between the weights array and an input float array."""
        result = 0
        if len(input) != len(self.getWeights()):
            raise ValueError('Input and Weights arrays must have the same dimension.')
        for i in range(len(input)):
            result += self.getWeights()[i] * input[i]
        return result

    def output(self, input):
        """Calculates a binary output according to the an input array and the attributes of the current Perceptron."""
        if self.dotProduct(input) + self.getBias() > 0:
            return 1
        else:
            return 0

    def increaseWeights(self, input, C):
        """Increase all the weights according to the input array values and a C constant."""
        if len(input) != len(self.getWeights()):
            raise ValueError('Input and Weights arrays must have the same dimension.')
        for i in range(len(input)):
            self.setWeight(i, self.getWeights()[i] + C * input[i])

    def decreaseWeights(self, input, C):
        """Decrease all the weights according to the input array values and a C constant."""
        if len(input) != len(self.getWeights()):
            raise ValueError('Input and Weights arrays must have the same dimension.')
        for i in range(len(input)):
            self.setWeight(i, self.getWeights()[i] - C * input[i])

    def learn(self, input, desiredOutput, C):
        """Modifies the weights depending on the difference between the output obtained by an input and a desired
           output value."""
        actualOutput = self.output(input)
        if actualOutput != desiredOutput:
            if actualOutput < desiredOutput:
                self.increaseWeights(input, C)
            else:
                self.decreaseWeights(input, C)
            return False
        else:
            return True