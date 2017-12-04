from genetic_programing import ASTNode


class DivNode(ASTNode.ASTNode):
    """ASTNode inherited class that implement the division operation for evaluation."""
    def __init__(self, left=None, right=None):
        """DivNode initializer."""
        super(DivNode, self).__init__(left, right, '%')

    def evaluate(self, values):
        """Returns the value of the protected division of the left and right ASTNodes evaluation."""
        rightEvaluation = self.right.evaluate(values)
        if rightEvaluation == 0:
            return 1
        else:
            return self.left.evaluate(values) / rightEvaluation