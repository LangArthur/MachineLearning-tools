#
# Created on Tue Feb 16 2021
#
# Arthur Lang
# Utilities.py
#

## sumColumn
# sum a column in an array
# @param array array with the column
# @param i index of the column to sum
def sumColumn(array, i):
    res = 0
    for elem in array:
        res += elem[i]
    return res

def is_float(val):
        try:
            float(val)
        except ValueError:
            return False
        else:
            return True