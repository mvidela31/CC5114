import copy
import random


class ASTNode(object):
    """Abstract class. Contains the basic methods of the AST nodes."""
    def __init__(self, left=None, right=None, symbol=None):
        """
        ASTNode initializer.
        :param left: Left ASTNode.
        :param right: Right ASTNode.
        :param symbol: String that represent the specific ASTNode subclass.
        :attr maxDepth: Integer that represent the max depth for growing.
        """
        self.left = left
        self.right = right
        self.symbol = symbol
        self.maxDepth = None

    def getMaxDepth(self):
        """Returns the current max depth from the current AST."""
        return self.maxDepth

    def getMinDepth(self):
        """Return the max depth of the lowest node from the current AST."""
        if issubclass(type(self), TerminalNode.TerminalNode):
            return self.maxDepth
        else:
            return min(self.left.getMinDepth(), self.right.getMinDepth())

    def getRandomNodeFromDepth(self, depth):
        """Returns a random node from the current AST with an specific depth."""
        if depth == self.maxDepth:
            return self
        if issubclass(type(self), TerminalNode.TerminalNode):
            return None
        if random.choice([True, False]):
            selectedLeft = self.left.getRandomNodeFromDepth(depth)
            if selectedLeft is not None:
                return selectedLeft
            selectedRight = self.right.getRandomNodeFromDepth(depth)
            if selectedRight is not None:
                return selectedRight
            return None
        else:
            selectedRight = self.right.getRandomNodeFromDepth(depth)
            if selectedRight is not None:
                return selectedRight
            selectedLeft = self.left.getRandomNodeFromDepth(depth)
            if selectedLeft is not None:
                return selectedLeft
            return None

    def getRandomNode(self):
        """"Returns a copy of a random node from the current AST."""
        selectedDepth = random.randint(self.getMinDepth(), self.maxDepth)
        selectedNode = self.getRandomNodeFromDepth(selectedDepth)
        return copy.deepcopy(selectedNode)

    def setRandomNodeFromDepth(self, node, depth):
        """Swap a random node from the current AST with an specific depth from another."""
        if depth == self.maxDepth:
            currentMaxDepth = self.maxDepth
            self = node
            self.updateDepth(currentMaxDepth)
            return self
        if issubclass(type(self), TerminalNode.TerminalNode):
            return None
        if random.choice([True, False]):
            selectedLeft = self.left.setRandomNodeFromDepth(node, depth)
            if selectedLeft is not None:
                self.left = selectedLeft
                return self
            selectedRight = self.right.setRandomNodeFromDepth(node, depth)
            if selectedRight is not None:
                self.right = selectedRight
                return self
            return None
        else:
            selectedRight = self.right.setRandomNodeFromDepth(node, depth)
            if selectedRight is not None:
                self.right = selectedRight
                return self
            selectedLeft = self.left.setRandomNodeFromDepth(node, depth)
            if selectedLeft is not None:
                self.left = selectedLeft
                return self
            return None

    def setRandomNode(self, node):
        """Swap a random node from the current AST from another bounded by the current AST max depth."""
        nodeDepth = node.getMaxDepth() - node.getMinDepth()
        selfMinDepth = self.getMinDepth()
        if selfMinDepth > nodeDepth:
            selectedDepth = random.randint(selfMinDepth, self.maxDepth)
        else:
            selectedDepth = random.randint(nodeDepth, self.maxDepth)
        newNode = self.setRandomNodeFromDepth(node, selectedDepth)
        return newNode

    def updateDepth(self, maxDepth):
        """Update the max depth of all the nodes from the current AST according to an initial max depth."""
        self.maxDepth = maxDepth
        if not issubclass(type(self), TerminalNode.TerminalNode):
            self.left.updateDepth(maxDepth - 1)
            self.right.updateDepth(maxDepth - 1)

    def evaluate(self, values):
        """Returns the numeric result of the evaluation of the current AST."""
        raise NotImplementedError("AST is an abstract class, implement the evaluate() method in a inherited class.")

    def randomGrowth(self, max_depth, nodes):
        """Growths the nodes of the current AST randomly bounded by a max depth."""
        if max_depth == 1:
            self.left = nodes.getRandomTerminalNode()
            self.right = nodes.getRandomTerminalNode()
            self.left.randomTerminalGrowth()
            self.right.randomTerminalGrowth()
        else:
            self.left = nodes.getRandomNode()
            if issubclass(type(self.left), TerminalNode.TerminalNode):
                self.left.randomTerminalGrowth()
            else:
                self.left.randomGrowth(max_depth - 1, nodes)
            self.right = nodes.getRandomNode()
            if issubclass(type(self.right), TerminalNode.TerminalNode):
                self.right.randomTerminalGrowth()
            else:
                self.right.randomGrowth(max_depth - 1, nodes)

    def printAST(self):
        """Prints the equation of an ASTNode."""
        print('(', end='')
        self.left.printAST()
        print(' ' + self.symbol + ' ', end='')
        self.right.printAST()
        print(')', end='')

from genetic_programing import TerminalNode