from neural_networks import NeuronLayer, OutputLayer, Neuron
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    """A double linked list object that contains multiple interconnected layers with neurons able to make response to an
       input signal and modifies its parameters to improve this response."""

    def __init__(self, networkArrayRepresentation):
        """
        :Attribute firstLayer: First element of a double linked list which contains an array of neurons.
        :param networkArrayRepresentation: An array of integers that represent the number of neurons of each layer.
        networkArrayRepresentation = [numberOfInputs, numberOfNeuronsOfHiddenLayer1, numberOfNeuronsOfHiddenLayer2,...,
        numberOfHiddenNeuronsn, numberOfNeuronsOfOutputLayer].
        """
        self.firstLayer = None
        self.insertLayersArray(self.getLayersArray(networkArrayRepresentation))

    def getLayersArray(self, networkArrayRepresentation):
        """Returns an array of neurons according to the networkArrayRepresentation adjusting the number of weights of
           neurons in each layer to the inputs that each layer will be receive."""
        layersArray = []
        for i in range(1, len(networkArrayRepresentation)):
            input = networkArrayRepresentation[i - 1]
            numberOfNeurons = networkArrayRepresentation[i]
            layerNeurons = []
            for j in range(numberOfNeurons):
                layerNeurons.append(Neuron.Neuron(input))
            layersArray.append(layerNeurons)
        return layersArray

    def insertLayersArray(self, layersArray):
        """Insert all de arrays of neurons in layersArrays linking each one to the first layer of the network."""
        layersArray.reverse()
        for layerArray in layersArray:
            self.addLayer(layerArray)

    def addLayer(self, layer):
        """Add a layer to the current neural network linking it to the first layer."""
        if self.firstLayer is None:
            self.firstLayer = OutputLayer.OutputLayer(layer, None)
        else:
            hiddenLayer = NeuronLayer.NeuronLayer(layer, None, None)
            hiddenLayer.nextLayer = self.firstLayer
            self.firstLayer.previousLayer = hiddenLayer
            self.firstLayer = hiddenLayer

    def getOutput(self):
        """Returns the output values of the current neural network."""
        return self.getOutputLayer().getOutput()

    def getOutputLayer(self):
        """Returns the output layer of the current neural network."""
        return self.firstLayer.getOutputLayer()

    def forwardFeed(self, input):
        """Feeds all the neural network with an input float array from the first layer to the output layer."""
        self.firstLayer.forwardFeed(input)

    def backwardPropagateError(self, desiredOutput):
        """Propagate the error obtained from the comparison between the current output of the neural network and the
           desired output from the output layer to the first layer."""
        self.getOutputLayer().backwardPropagateError(desiredOutput)

    def updateWeightsAndBias(self, input, learningRate):
        """Update the weights and bias of all the neurons in the neural network according to an input and a learning
           rate from the first layer to the output layer."""
        self.firstLayer.updateWeightsAndBias(input, learningRate)

    def train(self, input, desiredOutput, learningRate):
        """Modifies the parameters of all the neurons in the neural network to improve it answer to inputs signals."""
        self.forwardFeed(input)
        self.backwardPropagateError(desiredOutput)
        self.updateWeightsAndBias(input, learningRate)

    def prediction(self):
        """Return a prediction of the desired output according to the current output value."""
        if len(self.getOutputLayer().getNeurons()) > 1:
            return self.multipleOutputPrediction()
        else:
            return self.singleOutputPrediction()

    def singleOutputPrediction(self):
        """Return a prediction of the desired single output."""
        if self.getOutput()[0] > 0.5:
            return [1]
        else:
            return [0]

    def multipleOutputPrediction(self):
        """Returns a prediction of the desired output according to the max value of the current neural network output."""
        maxOutputValue = 0.0
        maxNeuronPosition = 0
        output = self.getOutput()
        for i in range(len(output)):
            if output[i] > maxOutputValue:
                maxOutputValue = output[i]
                maxNeuronPosition = i
        predictedOutput = [0]*len(output)
        predictedOutput[maxNeuronPosition] = 1
        return predictedOutput

    def learn(self, dataSet, epochs, learningRate):
        """Modifies the attributes of each neuron in the current neural network to adapt it output to the desired output
           and plots the learning curve of the current neural network showing the accuracy rate determined by
           the quotient between the good guesses and the number of attempts for each epoch, and the error rate
           determined by the Mean Squared Error between the current output and the desired output for each epoch."""
        epochsArray = []
        accuracyArray = []
        errorArray = []
        for i in range(epochs):
            numberOfAttempts = 0.0
            goodGuesses = 0.0
            squaredErrors = 0.0
            predictions = 0.0
            for j in range(len(dataSet)):
                self.train(dataSet[j][0], dataSet[j][1], learningRate)
                # Squared errors
                for k in range(len(dataSet[j][1])):
                    squaredErrors += (dataSet[j][1][k] - self.getOutput()[k]) ** 2
                    predictions += 1
                # Good guesses
                if self.prediction() == dataSet[j][1]:
                    goodGuesses += 1
                numberOfAttempts += 1
            accuracyRate = goodGuesses / numberOfAttempts
            MSE = squaredErrors / predictions
            epochsArray.append(i)
            accuracyArray.append(accuracyRate)
            errorArray.append(MSE)
        plt.plot(epochsArray, accuracyArray, label='Accuracy')
        plt.legend()
        plt.plot(epochsArray, errorArray, label='Error')
        plt.legend()
        plt.ylim(0, 1)
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Rate')
        plt.show()