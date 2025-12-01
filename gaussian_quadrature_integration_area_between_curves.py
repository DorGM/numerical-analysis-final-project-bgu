"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy
import numpy as np
from numpy import log, sin, cos, e, tan, arctan
import time
import random
import math


def intersections(f1: callable, f2: callable, a: float, b: float, maxerr=0.001):
    f = lambda x: f1(x) - f2(x)
    d = maxerr
    lst_of_initials = []
    lst_of_roots = []
    lst_of_real_roots = []
    list_of_steps = np.arange(a, b, 0.1)
    lst_of_y = []
    try:
        lst_of_y = f(list_of_steps)
    except:
        # new_f = np.vectorize(f)
        # lst_of_y = new_f(list_of_steps)
        lst_of_y = [f(i) for i in list_of_steps]
    for i in range(1, len(lst_of_y)):
        if (lst_of_y[i]) * (lst_of_y[i - 1]) < 0:
            lst_of_initials.append([list_of_steps[i - 1], list_of_steps[i]])
        elif (lst_of_y[i]) * (lst_of_y[i - 1]) == 0:
            if (lst_of_y[i - 1]) == 0:
                # lst_of_roots.append((list_of_steps[i - 1]))
                lst_of_roots = np.append(lst_of_roots, list_of_steps[i - 1])
            else:
                # lst_of_roots.append(list_of_steps[i])
                lst_of_roots = np.append(lst_of_roots, list_of_steps[i])
        elif abs(abs((lst_of_y[i])) - abs((lst_of_y[i - 1]))) < 10 * d and abs(lst_of_y[i]) < 10 * d and abs(
                lst_of_y[i - 1]) < 10 * d:
            lst_of_initials.append([list_of_steps[i - 1], list_of_steps[i]])
    if len(lst_of_initials) == 0:
        return []
    else:
        for i in lst_of_initials:
            a = min(i[0], i[1])
            b = max(i[0], i[1])
            z = (a + b) / 2
            temp = 0
            fz = f(z)
            while abs(fz) > d:
                fa = f(a)
                fb = f(b)
                if fa == fb:
                    break
                z = b - fb * (b - a) / (fb - fa)
                fz = f(z)
                if temp == 1000:
                    break
                if (a, b) == (b, a):
                    break
                if z < a or z > b:
                    z = 'BlaBla'
                    break
                elif fz * fb < 0:
                    a = z
                else:
                    b = z
                temp += 1
            if z != 'BlaBla' and abs(fz) < d:
                lst_of_roots = np.append(lst_of_roots, z)
        lst_of_roots = np.sort(lst_of_roots, axis=None)
        if len(lst_of_roots) == 0:
            return []
        return lst_of_roots


def getGaussParams(num):
    if num == 1:
        X = numpy.array([1])
        A = numpy.array([2])
    elif num == 2:
        X = numpy.array([numpy.math.sqrt(1 / 3), -numpy.math.sqrt(1 / 3)])
        A = numpy.array([1, 1])
    elif num == 3:
        X = numpy.array([numpy.math.sqrt(3 / 5), -numpy.math.sqrt(3 / 5), 0])
        A = numpy.array([5 / 9, 5 / 9, 8 / 9])
    elif num == 6:
        X = numpy.array(
            [0.238619186081526, -0.238619186081526, 0.661209386472165, -0.661209386472165, 0.932469514199394,
             -0.932469514199394])
        A = numpy.array(
            [0.467913934574257, 0.467913934574257, 0.360761573028128, 0.360761573028128, 0.171324492415988,
             0.171324492415988])
    elif num == 10:
        X = numpy.array(
            [0.973906528517240, -0.973906528517240, 0.433395394129334, -0.433395394129334, 0.865063366688893,
             -0.865063366688893, 0.148874338981367, -0.148874338981367, 0.679409568299053, -0.679409568299053])
        A = numpy.array(
            [0.066671344307672, 0.066671344307672, 0.269266719309847, 0.269266719309847, 0.149451349151147,
             0.149451349151147, 0.295524224714896, 0.295524224714896, 0.219086362515885, 0.219086362515885])
    else:
        raise Exception(">>> Unsupported num = {} <<<".format(num))
    return X, A


def getGaussQuadrature(func, a, b, N):
    val = 0
    coeff = [10, 6, 3, 2, 1]
    lst_j = []
    if N == 1:
        return 0
    else:
        def chanN(n):
            while n >= 1:
                for i in coeff:
                    if n >= i:
                        n -= i
                        lst_j.append(i)
                        break
            return lst_j

        lst_j = chanN(N)
        lin = numpy.linspace(a, b, len(lst_j) + 1)
        for i in range(len(lst_j)):
            X, A = getGaussParams(lst_j[i])
            a = lin[i]
            b = lin[i + 1]
            term1 = (b - a) / 2
            term2 = (a + b) / 2
            term3 = numpy.array([func(term1 * x + term2) for x in X])
            term4 = numpy.sum(A * term3)
            val += term1 * term4
        return val



class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> numpy.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        result = getGaussQuadrature(f, a, b, n)
        result = numpy.float32(result)
        return result

    def areabetween(self, f1: callable, f2: callable) -> numpy.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        ff = lambda x: abs(f1(x) - f2(x))
        lst_roots = intersections(f1, f2, 1, 100)

        surface = 0

        for i in range(len(lst_roots) - 1):
            surface += (getGaussQuadrature(ff, lst_roots[i], lst_roots[i + 1], 30))

        result = numpy.float32(surface)
        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
from functionUtils import *


class TestAssignment3(unittest.TestCase):

    # def test_integrate_float32(self):
    #     ass3 = Assignment3()
    #     f1 = numpy.poly1d([-1, 0, 1])
    #     r = ass3.integrate(f1, -1, 1, 10)
    #
    #     self.assertEquals(r.dtype, numpy.float32)

    # def test_integrate_hard_case(self):
    #     ass3 = Assignment3()
    #     # f1 = strong_oscilations()
    #     f1 = lambda x: log(x) - 100 * x
    #     r = ass3.integrate(f1, 1, 100, 50)
    #     true_result = -499588.483
    #     # print(abs(true_result-r))
    #     self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


    # def testarea(self):
    #     ass3 = Assignment3()
    #     # f1 = strong_oscilations()
    #     # f1=np.poly1d([-1, 0, 1])
    #     f1 = lambda x: (x**0.5) - 2
    #     f2 = lambda x: sin(x) + 5
    #     r = ass3.areabetween(f1, f2)
    #     # true_result = 7.78662 * 10 ** 33
    #     true_result= 15.8406
    #     # print(true_result)
    #     print("r=",r)
    #     print("err:",abs(r-true_result))
    #     self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    # def test_integrate_float32(self):
    #     ass3 = Assignment3()
    #     f1 = RESTRICT_INVOCATIONS(10)(np.poly1d([-1, 0, 1]))
    #     r = ass3.integrate(f1, -1, 1, 10)
    #     self.assertGreaterEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 211)
        true_result = -7.78662 * 10 ** 33
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = RESTRICT_INVOCATIONS(10)(np.poly1d([-1, 0, 1]))
        r = ass3.integrate(f1, -1, 1, 10)
        # print(abs(true_result - r))
        self.assertGreaterEqual(r.dtype, np.float32)

    def test_polynomial_and_ln_case(self):
        ass3 = Assignment3()
        f = RESTRICT_INVOCATIONS(15)(lambda x: (x - 4) * (x - 2.5) - math.log(x, math.e))
        r = ass3.integrate(f, 1, 5.5, 15)
        true_result = - (88 * math.log((11 / 2), math.e) - 153) / 16
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_case_constant(self):
        ass3 = Assignment3()
        f = RESTRICT_INVOCATIONS(7)(lambda x: 5)
        r = ass3.integrate(f, -5, 10, 7)
        true_result = 75
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_polynomial(self):
        ass3 = Assignment3()
        f = RESTRICT_INVOCATIONS(7)(lambda x: x * x - 3 * x + 5)
        r = ass3.integrate(f, -0.07, 7.1, 7)
        true_result = 79.546131
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_sin1(self):
        ass3 = Assignment3()
        f = lambda x: sin(x * x)
        r = ass3.integrate(f, -4.4, 7.31, 40)
        true_result = 1.221308543955602
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_exp1(self):
        ass3 = Assignment3()
        f = lambda x: math.e ** (-2 * x * x)
        r = ass3.integrate(f, -2.71, 10.42, 19)
        true_result = 1.253314099967343
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_arctan(self):
        ass3 = Assignment3()
        f = lambda x: arctan(x)
        r = ass3.integrate(f, -0.27, 5.63, 15)
        true_result = 6.074245208223067
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_sin2(self):
        ass3 = Assignment3()

        def f(x):
            return sin(x) / x

        r = ass3.integrate(f, -10.2, 8.4123, 20)
        true_result = 3.26705
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_ln1(self):
        ass3 = Assignment3()
        f = lambda x: 1 / math.log(x, math.e)
        r = ass3.integrate(f, 2.31, 9.74, 13)
        true_result = 4.60146
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_exp2(self):
        ass3 = Assignment3()
        f = lambda x: math.e ** (math.e ** x)
        r = ass3.integrate(f, 0, 4.2, 26)
        true_result = 1.39359 * (10 ** 27)
        # print(abs(true_result - r))
        # print(r)
        # print(true_result)
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_ln2(self):
        ass3 = Assignment3()
        f = lambda x: math.log(math.log(x, math.e), math.e)
        r = ass3.integrate(f, 2, 10, 15)
        true_result = 3.95291
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_sinln(self):
        ass3 = Assignment3()
        f = lambda x: math.sin(math.log(x, math.e))
        r = ass3.integrate(f, 3, 7, 13)
        true_result = 3.8853
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_strong_oscilations(self):
        ass3 = Assignment3()
        f = strong_oscilations()
        r = ass3.integrate(f, 1, 3, 9)
        true_result = 1.36924843371
        # print(abs(true_result - r))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_area_between_1(self):
        ass3 = Assignment3()
        f1, f2 = lambda x: -6 * (x - 7) ** 4 + 24 * x - 29, lambda x: x ** 2 - 10 * x - 5
        r = ass3.areabetween(f1, f2)
        true_result = 603.856
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_area_between_2(self):
        ass3 = Assignment3()
        f1, f2 = lambda x: -6 * (x - 7) ** 4 + 24 * x ** 3, lambda x: - (x ** 4) + 8 * x ** 3 - 20 * x ** 2 + 10 * x - 5
        r = ass3.areabetween(f1, f2)
        true_result = 1.35319 * 10 ** 6
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    #
    def test_area_between_3(self):
        ass3 = Assignment3()
        f1, f2 = lambda x: (x - 2) * (x - 5) * (x - 10), lambda x: (x - 2) * (x - 11)
        r = ass3.areabetween(f1, f2)
        true_result = 207 / 4 + 36 * math.sqrt(3)
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    #
    def test_area_between_4(self):
        ass3 = Assignment3()
        f1, f2 = lambda x: (x - 2) * (x - 5) * (x - 10) * (x - 15), lambda x: (x - 2) * (x - 11)
        r = ass3.areabetween(f1, f2)
        true_result = 2986.42
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    #
    def test_area_between_5(self):
        ass3 = Assignment3()
        f1, f2 = lambda x: (math.e ** x) / 37, lambda x: (x - 3) * (x - 4) * (x - 7)
        r = ass3.areabetween(f1, f2)
        true_result = 0.0095818
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    #
    def test_area_between_6(self):
        ass3 = Assignment3()
        f1, f2 = lambda x: (x - 2) * (x - 5) * (x - 10) * (x - 15), lambda x: (x - 2) * (x - 11)
        r = ass3.areabetween(f1, f2)
        true_result = 2986.42
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_area_between_7(self):
        ass3 = Assignment3()
        f1, f2 = lambda x: (math.e ** x) / 4, lambda x: 5 * (x - 2) * (x - 3) * (x - 3.5) * (x - 6) * (x - 20)
        r = ass3.areabetween(f1, f2)
        true_result = 967.785
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

if __name__ == "__main__":
    unittest.main()

