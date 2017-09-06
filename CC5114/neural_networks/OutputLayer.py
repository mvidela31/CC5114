from neural_networks import NeuronLayer

class OutputLayer(NeuronLayer.NeuronLayer):
    """A node object that inherits from NeuronLayer. Last layer linked with its nextLayer attribute setted as None."""
    def __init__(self, neurons, previousLayer):
        """
        :param neurons: An array which contains all the neurons of the current layer.
        :param previousLayer: The previous layer linked to the current layer.
        """
        super(OutputLayer, self).__init__(neurons, previousLayer, None)

    def backwardPropagateError(self, desiredOutput):
        """Recursive method that modifies the delta value attribute of all the neurons of this layer calculating the
           corresponding error according to a desired output."""
        for i in range(len(self.neurons)):
            error = desiredOutput[i] - self.neurons[i].getOutput()
            self.neurons[i].updateDelta(error)
        if self.previousLayer is not None:
            self.previousLayer.hiddenBackwardPropagateError()

