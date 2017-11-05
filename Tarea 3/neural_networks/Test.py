import unittest, random
from neural_networks import Perceptron, ANDPerceptron, ORPerceptron, NANDPerceptron, SummingBitGate, Neuron,\
    NeuronLayer, OutputLayer, NeuralNetwork, DataSetParser

class Test(unittest.TestCase):

    def test_perceptron(self):
        perceptron = Perceptron.Perceptron(2)
        perceptron.setWeight(0, -2)
        perceptron.setWeight(1, -2)
        perceptron.setBias(3)
        self.assertEqual(2, len(perceptron.getWeights()))
        self.assertEqual([-2, -2], perceptron.getWeights())
        self.assertEqual(3, perceptron.getBias())

    def test_ANDPerceptron(self):
        andPerceptron = ANDPerceptron.ANDPerceptron()
        self.assertEqual(1, andPerceptron.output([1, 1]))
        self.assertEqual(0, andPerceptron.output([1, 0]))
        self.assertEqual(0, andPerceptron.output([0, 1]))
        self.assertEqual(0, andPerceptron.output([0, 0]))

    def test_ORPerceptron(self):
        orPerceptron = ORPerceptron.ORPerceptron()
        self.assertEqual(1, orPerceptron.output([1, 1]))
        self.assertEqual(1, orPerceptron.output([1, 0]))
        self.assertEqual(1, orPerceptron.output([0, 1]))
        self.assertEqual(0, orPerceptron.output([0, 0]))

    def test_NANDPerceptron(self):
        nandPerceptron = NANDPerceptron.NANDPerceptron()
        self.assertEqual(0, nandPerceptron.output([1, 1]))
        self.assertEqual(1, nandPerceptron.output([1, 0]))
        self.assertEqual(1, nandPerceptron.output([0, 1]))
        self.assertEqual(1, nandPerceptron.output([0, 0]))

    def test_SummingBitGate(self):
        summingBitGate = SummingBitGate.SummingBitGate()
        summingBitGate.operate([1, 1])
        self.assertEqual(0, summingBitGate.getOutput())
        self.assertEqual(1, summingBitGate.getCarryBit())
        summingBitGate.operate([1, 0])
        self.assertEqual(1, summingBitGate.getOutput())
        self.assertEqual(0, summingBitGate.getCarryBit())
        summingBitGate.operate([0, 1])
        self.assertEqual(1, summingBitGate.getOutput())
        self.assertEqual(0, summingBitGate.getCarryBit())
        summingBitGate.operate([0, 0])
        self.assertEqual(0, summingBitGate.getOutput())
        self.assertEqual(0, summingBitGate.getCarryBit())

    def test_learningPerceptron(self):
        perceptronA = NANDPerceptron.NANDPerceptron()
        perceptronA.learn([1, 1], 1, 0.5)
        self.assertEqual([-1.5, -1.5], perceptronA.getWeights())
        perceptronB = ORPerceptron.ORPerceptron()
        perceptronB.learn([1, 1], 0, 0.25)
        self.assertEqual([0.75, 0.75], perceptronB.getWeights())

    def test_neuron(self):
        neuron = Neuron.Neuron(2)
        neuron.setWeight(0, 1)
        neuron.setWeight(1, 1)
        neuron.setBias(1)
        self.assertEqual([1, 1], neuron.getWeights())
        self.assertEqual(1, neuron.getBias())
        self.assertIsNone(neuron.getOutput())
        self.assertIsNone(neuron.getDelta())
        self.assertEqual(0.25, neuron.transferDerivative(0.5))
        neuron.updateOutput([-1, 0])
        self.assertEqual(0.5, neuron.getOutput())
        neuron.updateDelta(0.5)
        self.assertEqual(0.125, neuron.getDelta())
        neuron.updateWeights([1, 0], 0.1)
        self.assertEqual([1.0125, 1.0], neuron.getWeights())
        neuron.updateBias(0.5)
        self.assertEqual(1.0625, neuron.getBias())

    def test_layers(self):
        nn = NeuralNetwork.NeuralNetwork([1, 1, 1])
        self.assertIs(type(nn.firstLayer), NeuronLayer.NeuronLayer)
        self.assertIs(type(nn.firstLayer.nextLayer), OutputLayer.OutputLayer)
        nn.firstLayer.neurons[0].setWeight(0, 0.5)
        nn.firstLayer.neurons[0].setBias(1)
        nn.firstLayer.nextLayer.neurons[0].setWeight(0, 0.5)
        nn.firstLayer.nextLayer.neurons[0].setBias(1)
        self.assertEqual(nn.firstLayer.getNeurons()[0].getWeights(), [0.5])
        self.assertEqual(nn.firstLayer.getNeurons()[0].getBias(), 1)
        self.assertIsNone(nn.firstLayer.getPreviousLayer())
        self.assertIs(type(nn.firstLayer.getNextLayer()), OutputLayer.OutputLayer)
        nn.firstLayer.forwardFeed([1])
        self.assertAlmostEqual(nn.firstLayer.getOutput()[0], 0.8, 1)
        self.assertAlmostEqual(nn.firstLayer.getNextLayer().getOutput()[0], 0.8, 1)
        self.assertIs(type(nn.firstLayer.getOutputLayer()), OutputLayer.OutputLayer)
        nn.firstLayer.getOutputLayer().backwardPropagateError([1.5])
        self.assertAlmostEqual(nn.firstLayer.getNeurons()[0].getDelta(), 0.008, 3)
        self.assertAlmostEqual(nn.firstLayer.getNextLayer().getNeurons()[0].getDelta(), 0.1, 1)
        nn.firstLayer.updateWeightsAndBias([1], 2.5)
        self.assertAlmostEqual(nn.firstLayer.getNeurons()[0].getWeights()[0], 0.52, 2)
        self.assertAlmostEqual(nn.firstLayer.getNeurons()[0].getBias(), 1.02, 2)
        self.assertAlmostEqual(nn.firstLayer.getNextLayer().getNeurons()[0].getWeights()[0], 0.72, 2)
        self.assertAlmostEqual(nn.firstLayer.getNextLayer().getNeurons()[0].getBias(), 1.27, 2)

    def test_AND_neural_network(self):
        set = [[[1, 1], [1]], [[1, 0], [0]], [[0, 1], [0]], [[0, 0], [0]]]
        epochs = 2000
        learningRate = 0.5
        nn = NeuralNetwork.NeuralNetwork([2, 2, 1])
        for i in range(epochs):
            for j in range(len(set)):
                nn.train(set[j][0], set[j][1], learningRate)
        nn.forwardFeed([1, 1])
        self.assertAlmostEqual(1.0, nn.getOutput()[0], 0)
        nn.forwardFeed([1, 0])
        self.assertAlmostEqual(0.0, nn.getOutput()[0], 0)
        nn.forwardFeed([0, 1])
        self.assertAlmostEqual(0.0, nn.getOutput()[0], 0)
        nn.forwardFeed([0, 0])
        self.assertAlmostEqual(0.0, nn.getOutput()[0], 0)

    def test_OR_neural_network(self):
        set = [[[1, 1], [1]], [[1, 0], [1]], [[0, 1], [1]], [[0, 0], [0]]]
        epochs = 2000
        learningRate = 0.5
        nn = NeuralNetwork.NeuralNetwork([2, 2, 1])
        for i in range(epochs):
            for j in range(len(set)):
                nn.train(set[j][0], set[j][1], learningRate)
        nn.forwardFeed([1, 1])
        self.assertAlmostEqual(1.0, nn.getOutput()[0], 0)
        nn.forwardFeed([1, 0])
        self.assertAlmostEqual(1.0, nn.getOutput()[0], 0)
        nn.forwardFeed([0, 1])
        self.assertAlmostEqual(1.0, nn.getOutput()[0], 0)
        nn.forwardFeed([0, 0])
        self.assertAlmostEqual(0.0, nn.getOutput()[0], 0)

    def test_XOR_neural_network(self):
        set = [[[1, 1], [0]], [[1, 0], [1]], [[0, 1], [1]], [[0, 0], [0]]]
        epochs = 2000
        learningRate = 0.5
        nn = NeuralNetwork.NeuralNetwork([2, 3, 1])
        for i in range(epochs):
            for j in range(len(set)):
                nn.train(set[j][0], set[j][1], learningRate)
        nn.forwardFeed([1, 1])
        self.assertAlmostEqual(0.0, nn.getOutput()[0], 0)
        nn.forwardFeed([1, 0])
        self.assertAlmostEqual(1.0, nn.getOutput()[0], 0)
        nn.forwardFeed([0, 1])
        self.assertAlmostEqual(1.0, nn.getOutput()[0], 0)
        nn.forwardFeed([0, 0])
        self.assertAlmostEqual(0.0, nn.getOutput()[0], 0)

if __name__ == '__main__':
    unittest.main()