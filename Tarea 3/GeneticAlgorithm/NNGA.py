import random
from neural_networks import NeuralNetwork
from GeneticAlgorithm import GA


class NNGA(GA.GA):
    def __init__(self, numberOfNeurons, populationSize, selectionSizeRate, mutationRate):
        super(NNGA, self).__init__(numberOfNeurons, populationSize, selectionSizeRate, mutationRate)
        self.fitnessArray = None

    def randomGene(self):
        """Generate a random gene: Neuron values: [Weight, Bias]."""
        return random.choice([random.random(), random.random()])

    def evolveEndCondition(self):
        return None

    def fitnessFunction(self, populationElementPosition):
        return self.fitnessArray[populationElementPosition]

    def updateFitness(self, fitnessArray):
        self.fitnessArray = fitnessArray

    def constructNeuralNetwork(self, individualPos):
        individual = self.getPopulation()[individualPos]
        NN = NeuralNetwork.NeuralNetwork()

