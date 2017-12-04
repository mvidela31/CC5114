from genetic_programing import ASTNode


class AddNode(ASTNode.ASTNode):
    """ASTNode inherited class that implement the addition operation for evaluation."""
    def __init__(self, left=None, right=None):
        """AddNode initializer."""
        super(AddNode, self).__init__(left, right, '+')

    def evaluate(self, values):
        """Returns the value of the addition of the left and right ASTNodes evaluation."""
        return self.left.evaluate(values) + self.right.evaluate(values)