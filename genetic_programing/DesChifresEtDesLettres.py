import string
import random
from genetic_programing import GA


class DesChifresEtDesLettres(GA.GA):
    """Des Chifres Et Des Lettres game implementation."""
    def __init__(self, populationSize, selectionSizeRate, crossOverProb, mutationProb, maxDepth, maxGen, numbers, match):
        """DesChifresEtDesLettres initializer."""
        self.numbers = numbers
        self.match = match
        variables = random.sample(list(string.ascii_lowercase), len(numbers))
        super(DesChifresEtDesLettres, self).__init__(populationSize, selectionSizeRate, crossOverProb, mutationProb,
                                                     maxDepth, maxGen, None, variables)

    def fitnessFunction(self, populationElementPosition):
        """Abstract method. Returns a value according to the genes of an individual element from the population."""
        return abs(self.match - self.population[populationElementPosition][0].evaluate(self.numbers))

    def evolveEndCondition(self):
        """Abstract method. Establish the end condition of the genetic algorithm."""
        if self.fitnessFunction(0) <= 0:
            return True
        else:
            return False