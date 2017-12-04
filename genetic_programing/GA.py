import random
import copy
import time
import matplotlib.pyplot as plt
from genetic_programing import AST


class GA(object):
    """Abstract class that implements a genetic algorithm for AST with two abstracts methods that must be implemented in
       the inherited classes with specified the fitness function and the end condition."""
    def __init__(self, populationSize, selectionSizeRate, crossOverProb, mutationProb, maxDepth, maxGen, numeric,
                 variables):
        """
        GA initializer.
        :param populationSize: Size of the population.
        :param selectionSizeRate: A rate related to the size of the selection tournaments.
        :param crossOverProb: Probability to make a cross-over between two parents.
        :param mutationProb: Probability to make a mutation to an individual.
        :param maxDepth: Max depth of the AST from the population.
        :param maxGen: Max number of generations.
        :param numeric: Bounds of random values for NumericTerminalNodes from the AST.
        :param variables: Variables of the VariableTerminalNodes from the AST.
        :attr population: An array of the individuals from the population. Each individual is an array which its first
                          element is a list of its genes and the second element is a the fitness value.
                          Individual: [ASTNode, fitness value].
        :attr AST: An AST with specific ASTNodes that can generate new ASTNodes.
        """
        self.crossOverProb = crossOverProb
        self.mutationProb = mutationProb
        self.populationSize = populationSize
        self.selectionSizeRate = selectionSizeRate
        self.maxDepth = maxDepth
        self.maxGen = maxGen
        self.population = None
        self.AST = AST.AST(numeric, variables)

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

    def getVariables(self):
        """Return the variables string array."""
        return self.AST.getVariables()

    def randomSample(self):
        """Generate a random AST."""
        return self.AST.getRandomAST(self.maxDepth)

    def initializePopulation(self):
        """Initialize the population with random individuals according to the population size attribute value."""
        self.population = [0] * self.getPopulationSize()
        for i in range(self.populationSize):
            self.population[i] = [self.randomSample(), None]

    def fitnessFunction(self, populationElementPosition):
        """Abstract method. Returns a value according to the genes of an individual element from the population."""
        raise NotImplementedError("GA is an abstract class, implement the fitnessFunction() method in a inherited class.")

    def updatePopulationFitness(self):
        """Update the fitness value of each individual from the population by evaluating the fitness function on each
           individual element."""
        for i in range(self.getPopulationSize()):
            self.population[i][1] = self.fitnessFunction(i)

    def populationTotalFitness(self):
        """Sum the fitness value of each individual element from the population."""
        totalFitness = 0
        for i in range(self.getPopulationSize()):
            totalFitness += self.getPopulation()[i][1]
        return totalFitness

    def sortPopulation(self, population):
        """Sort the current population attribute according to its fitness values using the bubble sort algorithm."""
        k = len(population)
        while k > 1:
            i = 0
            for j in range(k - 1):
                if population[j][1] > population[j + 1][1]:
                    aux = population[j]
                    population[j] = population[j + 1]
                    population[j + 1] = aux
                    i = j + 1
            k = i

    def tournamentSelection(self, tournamentSize):
        """Select the fittest individual of a random selection of tournamentSize population elements."""
        tournamentPopulation = random.sample(self.population, tournamentSize)
        self.sortPopulation(tournamentPopulation)
        champion = copy.deepcopy(tournamentPopulation[0][0])
        return champion

    def crossOver(self, parentA, parentB, probability):
        """Mix the nodes from parentA and parentB according to a random mixing point."""
        if random.random() <= probability:
            child = copy.deepcopy(parentB)
            node = parentA.getRandomNode()
            child.setRandomNode(node)
            return child
        else:
            parent = copy.deepcopy(random.choice([parentA, parentB]))
            return parent

    def mutation(self, individual, probability):
        """Swap a node of an individual with another randomly created according to a random mixing point."""
        if random.random() <= probability:
            randomNode = self.AST.getRandomAST(self.maxDepth)
            return individual.setRandomNode(randomNode)
        else:
            return individual

    def reproduction(self):
        """Produces a new AST generation by the use of genetic operations over the current population."""
        evolvedPopulation = []
        tournamentSize = int(self.populationSize * self.selectionSizeRate)
        for i in range(self.populationSize):
            parentA = self.tournamentSelection(tournamentSize)
            parentB = self.tournamentSelection(tournamentSize)
            child = self.crossOver(parentA, parentB, self.crossOverProb)
            child = self.mutation(child, self.mutationProb)
            evolvedPopulation.append([child, None])
        self.population = evolvedPopulation

    def evolveEndCondition(self):
        """Abstract method. Establish the end condition of the genetic algorithm."""
        raise NotImplementedError("GA is an abstract class, implement the evolveEndCondition() method in a inherited class.")

    def fittestCurve(self, fittestArray):
        """Plots the fitness values of the fittest individual of each generation."""
        generations = list(range(1, len(fittestArray) + 1))
        plt.plot(generations, fittestArray)
        plt.title("Fittest Individual Per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()

    def run(self):
        """Executes the genetic algorithm returning the fittest AST of the last generation."""
        generations = 1
        fittestArray = []
        startTime = time.time()
        self.initializePopulation()
        while True:
            self.updatePopulationFitness()
            self.sortPopulation(self.population)
            fittestArray.append(self.fitnessFunction(0))
            print('Generation: ' + str(generations) + ' / Fittest fitness: ' + str(self.fitnessFunction(0)))
            if self.evolveEndCondition() or generations == self.maxGen:
                break
            self.reproduction()
            generations += 1
        elapsedTime = time.time() - startTime
        print('Elapsed time: ' + str(elapsedTime) + ' seconds.')
        self.fittestCurve(fittestArray)
        return self.population[0][0]