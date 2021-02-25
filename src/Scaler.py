#
# Created on Tue Feb 23 2021
#
# Arthur Lang
# Scaler.py
#

import numpy
import copy

class Scaler():

    def _normalize(self, array):
        maxVal = numpy.max(array)
        minVal = numpy.min(array)
        return (array - maxVal) / (maxVal - minVal)

    def normalize(self, array, axe):
        return numpy.apply_along_axis(self._normalize, axe, array)
