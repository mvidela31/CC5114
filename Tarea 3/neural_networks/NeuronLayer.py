class NeuronLayer(object):
    """A node object with an array of neurons and a link to a previous and a next neuron layer."""

    def __init__(self, neurons, previousLayer, nextLayer):
        """neurons: An array which contains all the neurons of the current layer.
           previousLayer: The previous layer linked to the current layer.
           nextLayer: The next layer linked to the current layer."""
        self.neurons = neurons
        self.previousLayer = previousLayer
        self.nextLayer = nextLayer

    def getNeurons(self):
        """Return an array with all the neurons of the current layer."""
        return self.neurons

    def getPreviousLayer(self):
        """Return the previous layer linked to the current layer."""
        return self.previousLayer

    def getNextLayer(self):
        """Return the next layer linked to the current layer."""
        return self.nextLayer

    def getOutput(self):
        """Returns an array with the outputs attributes of each neuron in the current layer."""
        output = []
        for i in range(len(self.neurons)):
            output.append(self.neurons[i].getOutput())
        return output

    def forwardFeed(self, input):
        """Recursive method that modifies the output value attribute of all the neurons of this layer according to an
           input and uses that output as an input for the next layer feeding."""
        layerOutput = []
        for i in range(len(self.neurons)):
            self.neurons[i].updateOutput(input)
            layerOutput.append(self.neurons[i].getOutput())
        if self.nextLayer is not None:
            self.nextLayer.forwardFeed(layerOutput)

    def hiddenBackwardPropagateError(self):
        """Recursive method that modifies the delta value attribute of all the neurons of this layer calculating the
           corresponding error according to the weights and delta values of the next layer."""
        for i in range(len(self.neurons)):
            error = 0.0
            for j in range(len(self.nextLayer.neurons)):
                error += self.nextLayer.neurons[j].getWeights()[i] * self.nextLayer.neurons[j].getDelta()
            self.neurons[i].updateDelta(error)
        if self.previousLayer is not None:
            self.previousLayer.hiddenBackwardPropagateError()

    def updateWeightsAndBias(self, input, learningRate):
        """Recursive method that modifies the weights and bias of all the neurons in the current layer according to
           an input and a learning rate and uses the output of this layer as an input for the next layer."""
        for i in range (len(self.neurons)):
            self.neurons[i].updateWeights(input, learningRate)
            self.neurons[i].updateBias(learningRate)
        if self.nextLayer is not None:
            self.nextLayer.updateWeightsAndBias(self.getOutput(), learningRate)

    def getOutputLayer(self):
        """Return the last layer of the linked layers."""
        if self.getNextLayer() is not None:
            return self.nextLayer.getOutputLayer()
        else:
            return self

    def getWeightFromNeuron(self, numberOfLayer, numberOfNeuron, numberOfWeight):
        """Return a weight from the current layer according to a position."""
        if numberOfLayer == 0:
            if numberOfWeight == -1:
                return self.neurons[numberOfNeuron].getBias()
            else:
                return self.neurons[numberOfNeuron].getWeights()[numberOfWeight]
        else:
            return self.nextLayer.getWeightFromNeuron(numberOfLayer - 1, numberOfNeuron, numberOfWeight)

    def getNeuronFromLayer(self, numberOfLayer, numberOfNeuron):
        """Return a neuron from the current layer according to a position."""
        if numberOfLayer == 0:
            return self.neurons[numberOfNeuron]
        else:
            return self.nextLayer.getNeuronFromLayer(numberOfLayer - 1, numberOfNeuron)

    def setWeightFromNeuron(self, weight, numberOfLayer, numberOfNeuron, numberOfWeight):
        """Set a weight from the current layer according to a position."""
        if numberOfLayer == 0:
            if numberOfWeight == -1:
                self.neurons[numberOfNeuron].setBias(weight)
            else:
                self.neurons[numberOfNeuron].setWeight(numberOfWeight, weight)
        else:
            self.nextLayer.setWeightFromNeuron(weight, numberOfLayer - 1, numberOfNeuron, numberOfWeight)

    def setNeuronFromLayer(self, neuron, numberOfLayer, numberOfNeuron):
        """Set a neuron from the current layer according to a position."""
        if numberOfLayer == 0:
            self.neurons[numberOfNeuron] = neuron
        else:
            self.nextLayer.setNeuronFromLayer(neuron, numberOfLayer - 1, numberOfNeuron)