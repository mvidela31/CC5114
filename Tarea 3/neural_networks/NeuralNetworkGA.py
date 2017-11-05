import copy
import random
import pickle
import matplotlib.pyplot as plt
from neural_networks import NeuralNetwork, Neuron


class NeuralNetworkGA(object):
    """Genetic algorithm implementation for neural networks individuals."""
    def __init__(self, neuralNetworkArchitecture, populationSize, selectionSizeRate, mutationRate):
        """
        Constructor.
        :param neuralNetworkArchitecture: Architecture of the neural networks individuals.
        :param populationSize: Size of the population.
        :param selectionSizeRate: A rate related to the number of selected individuals from the original population.
        :param mutationRate: Rate of mutation of selected individuals.
        :attr population: An array of the individuals from the population. Each individual is an array which its first
                          element is a list of its genes and the second element is a the fitness value.
                          Individual: [Neural Network, fitness value].
        :attr numberOfNeurons: Total number if neurons of the neural network architecture.
        :attr numberOfWeights: Total number of weights and bias of the neural network architecture.
        :attr fittest: Array of fitness value of the fittest individual of each generation.
        """
        self.neuralNetworkArchitecture = neuralNetworkArchitecture
        self.numberOfNeurons = sum(self.getNeuralNetworkArchitecture()[1:])
        self.numberOfWeights = self.totalWeights()
        self.mutationRate = mutationRate
        self.populationSize = populationSize
        self.selectionSizeRate = selectionSizeRate
        self.population = self.initializePopulation()
        self.fittest = []

    def getNeuralNetworkArchitecture(self):
        """Return the arquitecture array representation of the neural network from the population."""
        return self.neuralNetworkArchitecture

    def getPopulation(self):
        """Return the population array attribute."""
        return self.population

    def getPopulationSize(self):
        """Return the size of the population attribute."""
        return self.populationSize

    def getMutationRate(self):
        """Return the mutation rate value."""
        return self.mutationRate

    def getSelectionSizeRate(self):
        """Return the selection rate value."""
        return self.selectionSizeRate

    def getNumberOfNeurons(self):
        """Return the number of neurons of the neural network architecture from the population."""
        return self.numberOfNeurons

    def getNumberOfWeights(self):
        """Return the number of weights and bias of the neural network architecture from the population."""
        return self.numberOfWeights

    def getPopulationIndividuals(self):
        """Return an array of each individual genes array without the fitness value."""
        populationArray = [0] * self.getPopulationSize()
        for i in range(self.getPopulationSize()):
            populationArray[i] = self.getPopulation()[i][0]
        return populationArray

    def randomSample(self):
        """Generate a neural network with random weights and bias according to the specified architecture."""
        return NeuralNetwork.NeuralNetwork(self.getNeuralNetworkArchitecture())

    def initializePopulation(self):
        """Initialize the population with random neural networks according to the population size attribute value and
           the specified architecture."""
        population = []
        for i in range(self.populationSize):
            population.append([self.randomSample(), None])
        return population

    def totalWeights(self):
        """Return the sum of weights and biases of the current neural network individual from the population."""
        totalWeights = 0
        for i in range(len(self.getNeuralNetworkArchitecture()) - 1):
            # Sum of weights per layer.
            totalWeights += self.getNeuralNetworkArchitecture()[i] * self.getNeuralNetworkArchitecture()[i+1]
            # Sum of biases per layer.
            totalWeights += self.getNeuralNetworkArchitecture()[i+1]
        return totalWeights

    def updateFitness(self, fitnessArray):
        """Update the fitness value of each individual from the population according to an input sorted fitness array
           which represent the fitness of every member from the population."""
        if len(fitnessArray) != self.populationSize:
            raise ValueError('Fitness array must have the same number of elements as the population size.')
        for i in range(self.getPopulationSize()):
            self.population[i][1] = fitnessArray[i]

    def sortPopulation(self):
        """Sort the current population according to it fitness values using the bubble sort algorithm."""
        k = self.getPopulationSize()
        while k > 1:
            i = 0
            for j in range(k - 1):
                if self.getPopulation()[j][1] < self.getPopulation()[j + 1][1]:
                    aux = self.getPopulation()[j]
                    self.population[j] = self.getPopulation()[j + 1]
                    self.population[j + 1] = aux
                    i = j + 1
            k = i
        self.fittest.append(self.getPopulation()[0][1])

    def selection(self):
        """Select the fittest members of the population according to max population number specified in the selection
           size rate attribute."""
        selectedPopulationNumber = (int)(self.getSelectionSizeRate() * self.getPopulationSize())
        self.population = self.getPopulation()[0:selectedPopulationNumber]

    def randomSelectedPos(self):
        """Returns two different of selected individual positions from the population."""
        maxSelectedPos = (int)(self.getSelectionSizeRate() * self.getPopulationSize()) - 1
        positionA = random.randint(0, maxSelectedPos)
        positionB = positionA
        while positionA == positionB:
            positionB = random.randint(0, maxSelectedPos)
        return [positionA, positionB]

    def selectWeight(self, visitedWeights):
        """Returns a weight not visited position."""
        numberOfLayer = random.randint(0, len(self.getNeuralNetworkArchitecture()[1:]) - 1)
        numberOfNeuron = random.randint(0, self.getNeuralNetworkArchitecture()[numberOfLayer + 1] - 1)
        numberOfWeight = random.randint(-1, self.getNeuralNetworkArchitecture()[numberOfLayer] - 1)
        while [numberOfLayer, numberOfNeuron, numberOfWeight] in visitedWeights:
            numberOfLayer = random.randint(0, len(self.getNeuralNetworkArchitecture()[1:]) - 1)
            numberOfNeuron = random.randint(0, self.getNeuralNetworkArchitecture()[numberOfLayer + 1] - 1)
            numberOfWeight = random.randint(-1, self.getNeuralNetworkArchitecture()[numberOfLayer] - 1)
        return [numberOfLayer, numberOfNeuron, numberOfWeight]

    def selectNeuron(self, visitedNeurons):
        """Returns a neuron not visited position."""
        numberOfLayer = random.randint(0, len(self.getNeuralNetworkArchitecture()[1:]) - 1)
        numberOfNeuron = random.randint(0, self.getNeuralNetworkArchitecture()[numberOfLayer + 1] - 1)
        while [numberOfLayer, numberOfNeuron] in visitedNeurons:
            numberOfLayer = random.randint(0, len(self.getNeuralNetworkArchitecture()[1:]) - 1)
            numberOfNeuron = random.randint(0, self.getNeuralNetworkArchitecture()[numberOfLayer + 1] - 1)
        return [numberOfLayer, numberOfNeuron]

    def crossOverWeights(self, parentA, parentB):
        """Creates a new neural network by mixing random weights from parentA and parentB neuronal networks."""
        child = copy.deepcopy(parentA)
        totalWeights = self.getNumberOfWeights()
        visitedWeights = []
        while len(visitedWeights) < totalWeights / 2:
            weightPosition = self.selectWeight(visitedWeights)
            extractedWeight = parentB.getWeight(weightPosition[0], weightPosition[1], weightPosition[2])
            child.setWeight(extractedWeight, weightPosition[0], weightPosition[1], weightPosition[2])
            visitedWeights.append(weightPosition)
        return child

    def crossOverNeurons(self, parentA, parentB):
        """Creates a new neural network by mixing random neurons from parentA and parentB neuronal networks."""
        child = copy.deepcopy(parentA)
        totalNodes = self.getNumberOfNeurons()
        visitedNeurons = []
        while len(visitedNeurons) < totalNodes / 2:
            neuronPosition = self.selectNeuron(visitedNeurons)
            extractedNeuron = parentB.getNeuron(neuronPosition[0], neuronPosition[1])
            child.setNeuron(extractedNeuron, neuronPosition[0], neuronPosition[1])
            visitedNeurons.append(neuronPosition)
        return child

    def crossOver(self):
        """Creates new neural networks offspring members for the population by mixing random features of two random
           neural networks from the selected population."""
        while len(self.getPopulation()) < self.getPopulationSize():
            [positionA, positionB] = self.randomSelectedPos()
            nodeA = self.getPopulation()[positionA][0]
            nodeB = self.getPopulation()[positionB][0]
            child = self.crossOverWeights(nodeA, nodeB)
            self.population.append([child, None])

    def unbiasedMutateWeight(self, individual, numberOfLayer, numberOfNeuron, numberOfWeight):
        """Change an specific weight from an individual to another random one."""
        individual.setWeight(random.uniform(-5, 5), numberOfLayer, numberOfNeuron, numberOfWeight)

    def biasedMutateWeight(self, individual, numberOfLayer, numberOfNeuron, numberOfWeight):
        """Adds a random value between -1 and 1 to some specific weight from an individual."""
        actualWeight = individual.getWeight(numberOfLayer, numberOfNeuron, numberOfWeight)
        individual.setWeight(actualWeight + random.uniform(-1, 1), numberOfLayer, numberOfNeuron, numberOfWeight)

    def mutateWeights(self, individual):
        """Changes some random weights from an individual neural network from the population to a new random one."""
        totalMutations = (int)(self.getMutationRate() * self.getNumberOfWeights())
        visitedWeights = []
        while len(visitedWeights) < totalMutations:
            weightPosition = self.selectWeight(visitedWeights)  # [layerPos, neuronPos, weightPos], Bias: weightPos = -1
            #self.unbiasedMutateWeight(individual, weightPosition[0], weightPosition[1], weightPosition[2])
            self.biasedMutateWeight(individual, weightPosition[0], weightPosition[1], weightPosition[2])
            visitedWeights.append(weightPosition)

    def mutateNeurons(self, individual):
        """Changes some random neurons from an individual neural network from the population to a new random one."""
        totalMutations = (int)(self.getMutationRate() * self.getNumberOfNeurons())
        visitedNeurons = []
        while len(visitedNeurons) < totalMutations:
            neuronPosition = self.selectNeuron(visitedNeurons)  # [layerPos, neuronPos]
            newNeuron = Neuron.Neuron(self.getNeuralNetworkArchitecture()[neuronPosition[0]])
            individual.setNeuron(newNeuron, neuronPosition[0], neuronPosition[1])
            visitedNeurons.append(neuronPosition)

    def mutation(self):
        """Mutate the offspring neural networks elements from the population."""
        initPos = (int)(self.getSelectionSizeRate() * self.getPopulationSize())
        for i in range(initPos, self.getPopulationSize()):
            self.mutateWeights(self.getPopulation()[i][0])

    def evolve(self):
        """Evolve the current population according to it individual fitness values."""
        self.sortPopulation()
        self.selection()
        self.crossOver()
        self.mutation()

    def fittestCurve(self):
        """Plots the fitness values of the fittest individual of each generation."""
        generations = list(range(1, len(self.fittest) + 1))
        plt.plot(generations, self.fittest)
        plt.title("Fittest Individual Per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()

    def save(self, filename):
        """Save the current object in a binary file."""
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)