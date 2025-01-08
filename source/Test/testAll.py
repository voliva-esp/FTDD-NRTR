"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""


import unittest
from source.Test import testToArray, testAddTdd, testSimpleTNContraction, testSimulate, testSimulateSlicing, \
    testSimulateBackends, testTNtoCotInput, testTNContraction, testSlicingMethods


if __name__ == '__main__':
    suite = unittest.TestSuite()
    test_modules = [testToArray, testAddTdd, testTNtoCotInput, testSimpleTNContraction, testTNContraction,
                    testSimulate, testSimulateBackends, testSlicingMethods, testSimulateSlicing]
    for test_module in test_modules:
        suite.addTests(unittest.loader.findTestCases(test_module))

    result = unittest.TextTestRunner(verbosity=2).run(suite)
