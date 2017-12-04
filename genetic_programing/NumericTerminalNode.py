import random
from genetic_programing import TerminalNode


class NumericTerminalNode(TerminalNode.TerminalNode):
    """TerminalNode inherited class that implement the numeric terminal node."""
    def __init__(self, numeric=None, variables=None, value=None):
        """VariableTerminalNode initializer."""
        super(NumericTerminalNode, self).__init__(numeric, None)
        self.value = value
        self.symbol = str(value)

    def evaluate(self, values):
        """Returns the terminal value for evaluation."""
        return self.value

    def randomTerminalGrowth(self):
        """Set a random value for value attribute between bounds according to the numeric array attribute."""
        self.value = random.uniform(self.numeric[0], self.numeric[1])
        self.symbol = str(self.value)