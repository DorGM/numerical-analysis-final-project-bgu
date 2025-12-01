"""
In this assignment you should find the intersection points for two functions.
"""

import numpy
import math
from numpy import log, sin, cos, e, poly1d
import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        f = lambda x: f1(x) - f2(x)
        d = maxerr
        lst_of_initials = []
        lst_of_roots = []
        lst_of_real_roots = []
        list_of_steps = np.arange(a, b, 0.01)
        lst_of_y = np.array([f(i) for i in list_of_steps])
        for i in range(1, len(lst_of_y)):
            if (lst_of_y[i]) * (lst_of_y[i - 1]) < 0:
                lst_of_initials.append([list_of_steps[i - 1], list_of_steps[i]])
            elif (lst_of_y[i]) * (lst_of_y[i - 1]) == 0:
                if (lst_of_y[i - 1]) == 0:
                    lst_of_roots = np.append(lst_of_roots, list_of_steps[i - 1])
                else:
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
            f_roots = [f(i) for i in lst_of_roots]
            lst_of_roots = list(lst_of_roots)
            for i in range(len(lst_of_roots)):
                if i == 0:
                    lst_of_real_roots.append(lst_of_roots[i])
                else:
                    if round(lst_of_roots[i], 3) == round(lst_of_roots[i - 1], 3):
                        if abs(f((lst_of_roots[i] + lst_of_real_roots[-1]) / 2)) < 7 * d:
                            if abs(f_roots[i]) < abs(f_roots[lst_of_roots.index(lst_of_real_roots[-1])]):
                                lst_of_real_roots.remove(lst_of_real_roots[-1])
                                lst_of_real_roots.append(lst_of_roots[i])
                        else:
                            lst_of_real_roots.append(lst_of_roots[i])
                    elif round(lst_of_roots[i], 2) == round(lst_of_roots[i - 1], 2):
                        if abs(f((lst_of_roots[i] + lst_of_real_roots[-1]) / 2)) < 7 * d:
                            if abs(f_roots[i]) < abs(f_roots[lst_of_roots.index(lst_of_real_roots[-1])]):
                                lst_of_real_roots.remove(lst_of_real_roots[-1])
                                lst_of_real_roots.append(lst_of_roots[i])
                        else:
                            lst_of_real_roots.append(lst_of_roots[i])
                    elif round(lst_of_roots[i], 1) == round(lst_of_roots[i - 1], 1):
                        if abs(f((lst_of_roots[i] + lst_of_real_roots[-1]) / 2)) < 7 * d:
                            if abs(f_roots[i]) < abs(f_roots[lst_of_roots.index(lst_of_real_roots[-1])]):
                                lst_of_real_roots.remove(lst_of_real_roots[-1])
                                lst_of_real_roots.append(lst_of_roots[i])
                        else:
                            lst_of_real_roots.append(lst_of_roots[i])
                    elif round(lst_of_roots[i], 0) == round(lst_of_roots[i - 1], 0):
                        if abs(f((lst_of_roots[i] + lst_of_real_roots[-1]) / 2)) < 7 * d:
                            if abs(f_roots[i]) < abs(f_roots[lst_of_roots.index(lst_of_real_roots[-1])]):
                                lst_of_real_roots.remove(lst_of_real_roots[-1])
                                lst_of_real_roots.append(lst_of_roots[i])
                        else:
                            lst_of_real_roots.append(lst_of_roots[i])
                    else:
                        lst_of_real_roots.append(lst_of_roots[i])
            if len(lst_of_real_roots) == 0:
                return []
            lst_of_real_roots = sorted(lst_of_real_roots)
            # print(len(lst_of_real_roots))
            return lst_of_real_roots


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    # def test_sqr(self):
    #
    #     ass2 = Assignment2()
    #
    #     f1 = np.poly1d([-1, 3, 1])
    #     f2 = np.poly1d([1, 0, -1])
    #     for i in range(10):
    #         X = ass2.intersections(f1, f2, -6, 6)
    #         # print(X)
    #         for x in X:
    #             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
    #     f1 = lambda x: (x ** 0.5) - 2
    #     f2 = lambda x: sin(x) + 5
    #     for i in range(10):
    #         X = ass2.intersections(f1, f2, 0, 5)
    #         # print(36.043 - X[0], 36.38 - X[1], 41.438 - X[2], 43.572 - X[3], 47.25 - X[4])
    #         for x in X:
    #             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
    #     f1 = lambda x: log(x)**10
    #     f2 = lambda x: cos(x**2)
    #     for i in range(10):
    #         X = ass2.intersections(f1, f2, -6, 6)
    #         # print(X)
    #         for x in X:
    #             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
    #     f1 = lambda x: (x**2 * sin(x**2) - 2)**2
    #     f2 = lambda x: log(x)
    #
    #     for i in range(10):
    #         X = ass2.intersections(f1, f2, -6, 6)
    #         # print(X)
    #         for x in X:
    #             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


    def test_poly(self):

        ass2 = Assignment2()

        # f1, f2 = randomIntersectingPolynomials(1000)
        # f1 = lambda x: x**17 + sin(x**2) + math.log(100 * x**2)
        # f2 = lambda x: cos(x**2)

        f1 = lambda x: sin(x**2)
        f2 = lambda x: sin(math.log(x))
        # f1 = lambda x: cos(x**3)
        # f2 = lambda x: 0
        # f1 = np.poly1d([-1, 3, 1])
        # f2 = np.poly1d([1, 0, -1])
        # f1 = strong_oscilations()
        # f2 = lambda x: 0
        X = ass2.intersections(f1, f2, 1, 4)
        print(f"test2:{X}")

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))



if __name__ == "__main__":
    unittest.main()

# print(TestAssignment2.test_sqrt())




