import random
import time


class GA(object):
    """
    Abstract class that implements a genetic algorithm with two abstracts methods that must be implemented in the
    inherited classes with specified the fitness function and the end condition.
    """
    def __init__(self, numberOfGenes, populationSize, selectionSizeRate, mutationRate):
        """
        Constructor.
        :param numberOfGenes: Size of individuals in the population.
        :param populationSize: Size of the population.
        :param selectionSizeRate: A rate related to the number of selected individuals from the original population.
        :param mutationRate: Rate of mutation of selected individuals.
        :attr population: An array of the individuals from the population. Each individual is an array which its first
                          element is a list of its genes and the second element is a the fitness value.
                          Individual: [[genes], fitness value].
        """
        self.numberOfGenes = numberOfGenes
        self.mutationRate = mutationRate
        self.populationSize = populationSize
        self.selectionSizeRate = selectionSizeRate
        self.population = None

    def getPopulation(self):
        """
        Return the population array attribute.
        """
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

    def getNumberOfGenes(self):
        """Return the size of each individual array from the population."""
        return self.numberOfGenes

    def randomGene(self):
        """Abstract method. Generate a random gene value according to the inherit class that implement it."""
        raise NotImplementedError("GA is an abstract class, implement the randomGene() method in a inherited class.")

    def randomSample(self):
        """Generate an individual with random genes."""
        sample = [0] * self.getNumberOfGenes()
        for gene in range(self.getNumberOfGenes()):
            sample[gene] = self.randomGene()
        return sample

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
            #self.population[i][1] = self.fitnessFunction(self.getPopulation()[i][0])


    def populationTotalFitness(self):
        """Sum the fitness value of each individual element from the population."""
        totalFitness = 0
        for i in range(self.getPopulationSize()):
            totalFitness += self.getPopulation()[i][1]
        return totalFitness

    def sortPopulation(self):
        """Sort the current population attribute according to its fitness values using the bubble sort algorithm."""
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

    def normalizePopulationFitness(self):
        """Normalize each fitness value from the individuals elements of the population."""
        totalFitness = self.populationTotalFitness()
        for i in range(self.getPopulationSize()):
            self.population[i][1] = self.getPopulation()[i][1] / totalFitness

    def selectIndividual(self):
        """Select individuals elements from population according to a random value between 0 and 1 and the accumulated
           fitness value of each individual."""
        R = random.random()
        accumulatedFitness = 0
        for i in range(len(self.getPopulation())):
            accumulatedFitness += self.getPopulation()[i][1]
            if accumulatedFitness >= R:
                if i == 0:
                    return self.getPopulation()[i]
                else:
                    return self.getPopulation()[i - 1]

    def selectionV2(self):
        """Select the fittest members of the population according to max population number specified in the selection
           size rate attribute."""
        self.normalizePopulationFitness()
        selectedPopulationSize = (int)(self.getSelectionSizeRate() * self.getNumberOfGenes())
        selectedPopulation = []
        for i in range(selectedPopulationSize):
            selectedIndividual = self.selectIndividual()
            #while selectedIndividual in selectedPopulation:
            #    selectedIndividual = self.selectIndividual()
            selectedPopulation.append(selectedIndividual)
        self.population = selectedPopulation

    def selection(self):
        """Select the fittest members of the population according to max population number specified in the selection
           size rate attribute."""
        selectedPopulationNumber = (int)(self.getSelectionSizeRate() * self.getNumberOfGenes())
        self.population = self.getPopulation()[0:selectedPopulationNumber]

    def populationWithFitneesToPopulation(self):
        """Return an array of each individual genes array without the fitness value."""
        populationArray = [0] * self.getPopulationSize()
        for i in range(self.getPopulationSize()):
            populationArray[i] = self.getPopulation()[i][0]
        return populationArray

    def generateChild(self, parentA, parentB):
        """Mix the genes from parentA and parentB according to a randomly mixing point."""
        mixingPoint = random.randint(0, len(parentA))
        return parentA[0:mixingPoint] + parentB[mixingPoint:len(parentB)]

    def crossOver(self):
        """Mix the genes some selected individuals changing the current population attribute to an array of mixed
           individuals."""
        newPopulation = []
        newPopulationSize = 0
        while newPopulationSize < self.getPopulationSize():
            positionA = random.randint(0, len(self.getPopulation()) - 1)
            positionB = positionA
            while positionA == positionB:
                positionB = random.randint(0, len(self.getPopulation()) - 1)
            newPopulation.append([self.generateChild(self.getPopulation()[positionA][0], self.getPopulation()[positionB][0]), None])
            newPopulationSize += 1
        self.population = newPopulation

    def mutateElement(self, element):
        """Changes some genes of an individual element randomly."""
        totalMutations = (int)(self.getMutationRate() * len(element))
        currentMutations = 0
        mutatedGenes = []
        while (currentMutations < totalMutations):
            geneToMutate = random.randint(0, self.getNumberOfGenes() - 1)
            if (geneToMutate not in mutatedGenes):
                element[geneToMutate] = self.randomGene()
                mutatedGenes.append(geneToMutate)
                currentMutations += 1
        return element

    def mutation(self):
        """Mutate the individuals element from the population."""
        for i in range(self.getPopulationSize()):
            self.population[i][0] = self.mutateElement(self.getPopulation()[i][0])

    def reproduction(self):
        """Cross the genes of the individuals from the population and changes some genes of it randomly."""
        self.crossOver()
        self.mutation()

    def evolveEndCondition(self):
        """Abstract method. Establish the end condition of the genetic algorithm."""
        raise NotImplementedError("GA is an abstract class, implement the evolveEndCondition() method in a inherited class.")

    def run(self):
        """Executes the genetic algorithm."""
        generations = 1
        startTime = time.time()
        self.initializePopulation()
        while True:
            self.updatePopulationFitness()
            self.sortPopulation()
            print("Current fittets: " + str(self.getPopulation()[0]))
            if self.evolveEndCondition():
                break
            self.selection()
            self.reproduction()
            generations += 1
        elapsedTime = time.time() - startTime
        print("Generations: " + str(generations))
        print("Elapsed time: " + str(elapsedTime))