"""
In this assignment you should interpolate the given function.
"""

import numpy
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import cos, sin, tan, arctan, log, e


def intersections(f1: callable, f2: callable, a: float, b: float, maxerr=0.001):
    d = maxerr
    f = lambda x: f1(x) - f2(x)
    z = (a + b) / 2
    fz = 1
    while abs(fz) > d:
        fa = f(a)
        fb = f(b)
        if fa == fb:
            break
        z = b - fb * (b - a) / (fb - fa)
        fz = f(z)
        # if temp == 1000:
        #     break
        # if (a, b) == (b, a):
        #     break
        if fz * fb < 0:
            a = z
        else:
            b = z
    if 1 - z < 0.001:
        z = 1
    elif z < 0.001:
        z = 0
    return z


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        x_axis_length = abs(b - a)
        delta = 0.0001
        # temp_array = np.array([1, 2, 3])
        # try:
        #     f(temp_array)
        #     indication += 1
        # except:
        #     indication = 0
        # if indication == 1:
        #     if abs(b - a) <= 5:
        #         n = int(1000 * abs(b - a))
        #     elif abs(b - a) <= 10:
        #         n = int(750 * abs(b - a))
        #     elif abs(b - a) <= 30:
        #         n = int(500 * abs(b - a))
        #     elif abs(b - a) <= 60:
        #         n = int(350 * abs(b - a))
        #     elif abs(b - a) <= 100:
        #         n = int(100 * abs(b - a))
        #     else:
        #         n = int(70 * abs(b - a))
        #     if (n - 1) % 3 != 0:
        #         if n % 3 == 0:
        #             n += 1
        #         else:
        #             n += 2
        #     linE = np.linspace(a, b, n, retstep=True)
        #     x_points = linE[0]
        #     spaces_between_Xs = linE[1]
        #     y_points = f(x_points)
        # else:
        #     n -= 1
        # if (n - 1) % 3 != 0:
        #     if n % 3 == 0:
        #         n -= 2
        #     else:
        #         n -= 1
        linE = np.linspace(a, b, n, retstep=True)
        x_points = linE[0]
        spaces_between_Xs = linE[1]
        # y_points = f(x_points)
        y_points = np.array([f(i) for i in x_points])

        def TDMAsolver(a, b, c, d):

            nf = len(d)  # number of equations
            ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
            for it in range(1, nf):
                mc = ac[it - 1] / bc[it - 1]
                bc[it] = bc[it] - mc * cc[it - 1]
                dc[it] = dc[it] - mc * dc[it - 1]

            xc = bc
            xc[-1] = dc[-1] / bc[-1]

            for il in range(nf - 2, -1, -1):
                xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

            return xc

        third = np.ones(len(x_points) - 2)
        sec = np.insert(np.insert(4 * np.ones(len(x_points) - 3), 0, 2), len(x_points) - 2, 7)
        first = np.insert(np.ones(len(x_points) - 3), len(x_points) - 3, 2)
        points = np.concatenate(([x_points], [y_points]), axis=0).T
        K = points[0] + 2 * points[1]
        K = np.append([K], 4 * points[1:len(x_points) - 2] + 2 * points[2:len(x_points) - 1], axis=0)
        K = np.append(K, [8 * points[len(x_points) - 2] + points[len(x_points) - 1]], axis=0)
        ax = TDMAsolver(first, sec, third, K[:, 0])
        ay = TDMAsolver(first, sec, third, K[:, 1])
        a1 = np.concatenate(([ax], [ay]), axis=0).T
        b1 = 2 * points[1:len(points) - 1] - a1[1:len(points) - 1]
        b1 = np.append(b1, [(a1[len(points) - 2] + points[len(points) - 1]) / 2], axis=0)

        dict_num_bezier_lambda = {}

        def get_num_bezier(x):
            x_points_lst = list(x_points.copy())
            x -= a
            k = abs(x / spaces_between_Xs)
            if int(k) == x_points_lst.index(x_points_lst[-1]):
                k -= 1
            k = int(k)

            if k not in dict_num_bezier_lambda.keys():
                dict_num_bezier_lambda[k] = [lambda t: (1 - t)**3 * x_points[k] + 3 * t * (1 - t)**2 * a1[k][0]
                                                        + 3 * t**2 * (1 - t) * b1[k][0] + t**3 * x_points[k + 1],
                                             lambda t: (1 - t)**3 * y_points[k] + 3 * t * (1 - t)**2 * a1[k][1]
                                                        + 3 * t**2 * (1 - t) * b1[k][1] + t**3 * y_points[k + 1]]
            return list(dict_num_bezier_lambda.get(k))

        def GetY(x):
            def GetT(x):
                numBezier = get_num_bezier(x)
                root_of_t = intersections(numBezier[0], lambda t: x, 0, 1.01)
                return numBezier[1](root_of_t)
            return GetT(x)
        return GetY

##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)
            # f = lambda x: math.log(math.log(x, 10), 10)
            # f = lambda x: math.log(math.log(x, 10), 10)
            # f = lambda x: cos(x ** 2) - x ** 7 + tan(x ** 5)
            # f = lambda x: x**sin(x**2) + e**x - cos(x**3)
            # f = lambda x: math.log(x, math.e)
            # f = lambda x: math.sin(x)
            ff = ass1.interpolate(f, -10, 10, 5)
            # n = [1, 10, 25, 50, 100, 200, 500, 1000]
            # xs = np.linspace(-9.9, 9.9, 200)
            # xs = np.random.random(200)
            # xs = np.random.uniform(low=5, high=10, size=200)
            xs = [-10, 10]
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)
    #
    # def test_with_poly_restrict(self):
    #     ass1 = Assignment1()
    #     a = np.random.randn(5)
    #     f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
    #     ff = ass1.interpolate(f, -10, 10, 10)
    #     xs = np.random.random(20)
    #     for x in xs:
    #         yy = ff(x)

    # def test_with_poly(self):
    #     T = time.time()
    #
    #     ass1 = Assignment1()
    #     mean_err = 0
    #
    #     d = 2
    #     for i in tqdm(range(100)):
    #         a = np.random.randn(d)
    #
    #         f = np.poly1d(a)
    #
    #         ff = ass1.interpolate(f, -10, 10, 100)
    #
    #         xs = np.random.random(200)
    #         err = 0
    #         for x in xs:
    #             yy = ff(x)
    #             y = f(x)
    #             err += abs(y - yy)
    #
    #         err = err / 200
    #         mean_err += err
    #     mean_err = mean_err / 100
    #
    #     T = time.time() - T
    #     print(T)
    #     print(mean_err)


if __name__ == "__main__":
    unittest.main()












