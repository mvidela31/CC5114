from genetic_programing import ASTNode


class MultNode(ASTNode.ASTNode):
    """ASTNode inherited class that implement the multiplication operation for evaluation."""
    def __init__(self, left=None, right=None):
        """MultNode initializer."""
        super(MultNode, self).__init__(left, right, '*')

    def evaluate(self, values):
        """Returns the value of the multiplication of the left and right ASTNodes evaluation."""
        return self.left.evaluate(values) * self.right.evaluate(values)