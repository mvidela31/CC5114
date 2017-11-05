from GeneticAlgorithm import BitGA, WordGA

# Parameters
populationSize = 10
selectedPopulationRate = 0.5
mutationRate = 0.25

# Bits sequence
bitSequence = [0,0,1,0,0]
#bitsGA = BitGA.BitGA(bitSequence, populationSize, selectedPopulationRate, mutationRate)
#bitsGA.run()

# Letters sequence
letterSquence = ['m','i','g','u','e','l']
wordGA = WordGA.WordGA(letterSquence, populationSize, selectedPopulationRate, mutationRate)
wordGA.run()