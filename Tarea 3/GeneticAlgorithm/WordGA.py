import random
import string
from GeneticAlgorithm import GA


class WordGA(GA.GA):
    def __init__(self, secretWord, populationSize, selectionSizeRate, mutationRate):
        super(WordGA, self).__init__(len(secretWord), populationSize, selectionSizeRate, mutationRate)
        self.secretWord = secretWord

    def randomGene(self):
        """Generate a random gene: String ASCII lowercase letter."""
        return random.choice(string.ascii_lowercase)

    def evolveEndCondition(self):
        """Establish the end condition of the genetic algorithm:
           The fitness value from the fittest individual equals to the size of the individuals genes"""
        if self.getPopulation()[0][1] == self.getNumberOfGenes():
            return True
        else:
            return False

    def fitnessFunction(self, populationElement):
        """Returns a value according to the genes of an individual element from the population.
           Return the number of equals elements in the same position of the individual popúlation element and the
           secret desired sequence.
        """
        fitness = 0
        for i in range(len(populationElement)):
            if populationElement[i] == self.secretWord[i]:
                fitness += 1
        return fitness