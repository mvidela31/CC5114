from neural_networks import NANDPerceptron

class SummingBitGate(object):
    """Object that calculates the sum between two bits, saving de operation result in the output attribute field and
       the carryBit attribute field."""
    def __init__(self, output=None, carryBit=None):
        """
        :param output: Saves the output bit after operate with an input.
        :param carryBit: Saves the carry bit after operate with an input.
        """
        self.output = output
        self.carryBit = carryBit

    def getOutput(self):
        """Return the output attribute."""
        return self.output

    def getCarryBit(self):
        """Return the carry bit attribute."""
        return self.carryBit

    def operate(self, input):
        """Refresh the output and carryBit attributes fields according to the input. The input must be an array
           of integers (0 or 1) of size 2."""
        NAND = NANDPerceptron.NANDPerceptron()
        output0 = NAND.output(input)
        input1 = [input[0], output0]
        output1 = NAND.output(input1)
        input2 = [input[1], output0]
        output2 = NAND.output(input2)
        input3 = [output1, output2]
        output3 = NAND.output(input3)
        self.output = output3
        input4 = [output0, output0]
        output4 = NAND.output(input4)
        self.carryBit = output4