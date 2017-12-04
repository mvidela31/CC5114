from genetic_programing import ASTNode


class SubNode(ASTNode.ASTNode):
    """ASTNode inherited class that implement the subtraction operation for evaluation."""
    def __init__(self, left=None, right=None):
        """SubNode initializer."""
        super(SubNode, self).__init__(left, right, '-')

    def evaluate(self, values):
        """Returns the value of the subtraction of the left and right ASTNodes evaluation."""
        return self.left.evaluate(values) - self.right.evaluate(values)