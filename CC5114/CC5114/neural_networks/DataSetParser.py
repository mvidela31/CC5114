class DataSetParser(object):
    """Object able to parse and adapt a data set from a text file to be used for neural network learning."""

    def __init__(self, fileName, lineDataSize):
        """fileName: Name of text file to parse.
           lineDataSize: Number of desired data in each line of the text file.
           adaptedData: Adapted data from text file parsing for a neural network."""
        self.fileName = fileName
        self.lineDataSize = lineDataSize
        self.adaptedData = None

    def getAdaptedData(self):
        """Returns the adapted data attribute."""
        return self.adaptedData

    def floatArrayData(self):
        """Converts the text file data into an array of floats arrays which contains the data of each text line."""
        floatArrayData = []
        file = open(self.fileName, 'r')
        for line in file:
            stringArray = line.split()
            if len(stringArray) == self.lineDataSize:
                floatArrayData.append(self.convertToFloatArray(stringArray))
        file.close()
        return floatArrayData

    def adaptData(self):
        """Adapts the text file data into an array wich contains a float array with the input values normalized and another
           array with the output value codified for every text line."""
        adaptedData = []
        floatArrayData = self.floatArrayData()
        maxInputValue, minInputValue, maxOutputValue, minOutputValue = self.getDataLimitValues(floatArrayData)
        for lineValues in floatArrayData:
            adaptedLineValues = []
            lineInputValues = lineValues[0:self.lineDataSize - 1]
            lineOutputValue = lineValues[self.lineDataSize - 1]
            adaptedLineInputValue = self.normalizeInputValues(lineInputValues, maxInputValue, minInputValue, 0, 1)
            # adaptedLineInputValue = lineInputValues
            adaptedLineOutputValue = self.adaptOutputValue(lineOutputValue, maxOutputValue, minOutputValue)
            adaptedLineValues.append(adaptedLineInputValue)
            adaptedLineValues.append(adaptedLineOutputValue)
            adaptedData.append(adaptedLineValues)
        self.adaptedData = adaptedData

    def convertToFloatArray(self, array):
        """Converts an array of string into an array of floats."""
        float_array = []
        for element in array:
            float_array.append(float(element))
        return float_array

    def normalizeValue(self, value, maxValue, minValue, normalizationHighLimit, normalizationLowLimit):
        """Normalize a value according to a maximum value, a minimum value and a normalization limit values."""
        return (value - minValue) * (normalizationHighLimit - normalizationLowLimit) / (maxValue - minValue) + normalizationLowLimit

    def normalizeInputValues(self, values, maxValue, minValue, normalizationHighLimit, normalizationLowLimit):
        """Normalize every value of an array according to a maximum value, a minimum value and a normalization limit values."""
        normalizedValues = []
        for value in values:
            normalizedValues.append(self.normalizeValue(value, maxValue, minValue, normalizationHighLimit, normalizationLowLimit))
        return normalizedValues

    def adaptOutputValue(self, outputValue, maxOutputValue, minOutputValue):
        """Codifies an integer value according to an integer maximum and minimum values."""
        adaptedOutput = [0] * int(maxOutputValue - minOutputValue + 1)
        adaptedOutput[int(outputValue) - 1] = 1
        return adaptedOutput

    def getDataLimitValues(self, floatArrayData):
        """Extract the maximum and minimum inputs and outputs values of a float array data set."""
        maxInputData = []
        minInputData = []
        outputData = []
        for values in floatArrayData:
            maxInputData.append(max(values[0:self.lineDataSize - 1]))
            minInputData.append(min(values[0:self.lineDataSize - 1]))
            outputData.append(values[self.lineDataSize - 1])
        return [max(maxInputData), min(minInputData), max(outputData), min(outputData)]