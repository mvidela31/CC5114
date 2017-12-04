from genetic_programing import ASTNode


class TerminalNode(ASTNode.ASTNode):
    """ASTNode inherited class that implement the terminal node which have a value for evaluation."""
    def __init__(self, numeric=None, variables=None, symbol=None):
        """
        TerminalNode initialization.
        :param numeric: Bounds of random values for NumericTerminalNodes from the AST. Ex: [minVal, maxVal].
        :param variables: Variables of the VariableTerminalNodes from the AST. Ex: ['var1', 'var2',..., 'varN'].
        :param symbol: String of the value of the terminal node.
        """
        self.numeric = numeric
        self.variables = variables
        self.symbol = symbol

    def printAST(self):
        """Print the symbol of the terminal node."""
        print(self.symbol, end='')