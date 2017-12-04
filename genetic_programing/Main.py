from genetic_programing import DesChifresEtDesLettres

# Parameters.
populationSize = 50
selectionSizeRate = 0.3
crossOverProb = 0.9
mutationProb = 0.1
maxDepth = 4
maxGen = 50
numberSet = [27, 7, 8, 100, 4, 2]
match = 459

# Game.
game = DesChifresEtDesLettres.DesChifresEtDesLettres(populationSize, selectionSizeRate, crossOverProb, mutationProb,
                                                     maxDepth, maxGen, numberSet, match)
result = game.run()

# Prints.
print()
print('Result:')
result.printAST()
print()
print()
print('Symbology:')
print(str(numberSet) + ' = ' + str(game.getVariables()))