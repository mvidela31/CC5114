import random
from GeneticAlgorithm import GA


class BitGA(GA.GA):

    def __init__(self, secretBits, populationSize, selectionSizeRate, mutationRate):
        super(BitGA, self).__init__(len(secretBits), populationSize, selectionSizeRate, mutationRate)
        self.secretBits = secretBits

    def randomGene(self):
        """Generate a random gene: Bit (0 or 1)"""
        return random.randint(0, 1)

    def evolveEndCondition(self):
        """Establish the end condition of the genetic algorithm:
           The fitness value from the fittest individual equals to the size of the individuals genes"""
        if self.getPopulation()[0][1] == self.getNumberOfGenes():
            return True
        else:
            return False

    def fitnessFunction(self, populationElement):
        """Returns a value according to the genes of an individual element from the population.
           Return the number of equals elements in the same position of the individual population element and the
           secret desired sequence.
        """
        fitness = 0
        for i in range(len(populationElement)):
            if populationElement[i] == self.secretBits[i]:
                fitness += 1
        return fitness