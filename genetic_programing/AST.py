import copy
import random
from genetic_programing import AddNode, SubNode, MultNode, DivNode, NumericTerminalNode, VariableTerminalNode


class AST(object):
    """Abstract Syntax Tree Implementation."""
    def __init__(self, numeric, variables):
        """
        AST intializer. If one of the parametes are None, the terminal nodes generates will not have these parameters.
        :param numeric: [min, max]. Defines the bounds of the numeric terminal nodes values.
        :param variables: ['v1','v2',...,'vn']. Defines the variables of the variables terminal nodes.
        """
        self.variables = variables
        self.operationNodes = [AddNode.AddNode(), SubNode.SubNode(), MultNode.MultNode(), DivNode.DivNode()]
        self.terminalNodes = None
        self.updateTerminalNodes(numeric, variables)

    def getVariables(self):
        """Returns the variables string array."""
        return self.variables

    def getRandomNode(self):
        """Returns a random node."""
        return copy.deepcopy(random.choice(self.operationNodes + self.terminalNodes))

    def getRandomOperationNode(self):
        """Returns a random operation node."""
        return copy.deepcopy(random.choice(self.operationNodes))

    def getRandomTerminalNode(self):
        """Returns a random terminal node."""
        return copy.deepcopy(random.choice(self.terminalNodes))

    def updateTerminalNodes(self, numeric, variables):
        """Update the terminal nodes according to the numeric and variables parameters."""
        if numeric is not None and variables is not None:
            self.terminalNodes = [NumericTerminalNode.NumericTerminalNode(numeric, variables),
                                  VariableTerminalNode.VariableTerminalNode(numeric, variables)]
        elif numeric is not None:
            self.terminalNodes = [NumericTerminalNode.NumericTerminalNode(numeric, variables)]
        elif variables is not None:
            self.terminalNodes = [VariableTerminalNode.VariableTerminalNode(numeric, variables)]
        elif numeric is None and variables is None:
            raise ValueError('At least one of the parameters must be different than None.')

    def getRandomAST(self, max_depth):
        """Return a random AST with deep bounded by a max depth."""
        root = self.getRandomOperationNode()
        root.randomGrowth(max_depth, self)
        root.updateDepth(max_depth)
        return copy.deepcopy(root)