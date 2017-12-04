import random
from genetic_programing import TerminalNode


class VariableTerminalNode(TerminalNode.TerminalNode):
    """ASTNode inherited class that implement the division operation for evaluation."""
    def __init__(self, numeric=None, variables=None, symbol=None):
        """VariableTerminalNode initializer."""
        super(VariableTerminalNode, self).__init__(None, variables)
        self.value = None
        self.symbol = symbol

    def evaluate(self, values):
        """Update the value of the current variable according to an input values array."""
        if len(values) != len(self.variables):
            raise ValueError('Incorrect input values array size. Variables to evaluate: ' + str(self.variables) + '.')
        for i in range(len(self.variables)):
            if self.variables[i] == self.symbol:
                return values[i]

    def randomTerminalGrowth(self):
        """Set a random variable for terminal attribute."""
        self.symbol = random.choice(self.variables)