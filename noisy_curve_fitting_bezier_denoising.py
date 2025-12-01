def intersections(f1: callable, f2: callable, a: float, b: float, maxerr=0.001):
    d = maxerr
    f = lambda x: f1(x) - f2(x)
    z = (a + b) / 2
    temp = 0
    while abs(f(z)) > d:
        fa = f(a)
        fb = f(b)
        if fb == fa:
            break
        z = b - float(f(b) * (b - a) / (f(b) - f(a)))
        if f(z) * f(b) < 0:
            a = z
        else:
            b = z
        temp += 1
    if 1 - z < 0.01:
        z = 1
    elif z < 0.01:
        z = 0
    return z



import numpy as np
import time
import random
import sys
import math


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        def interpolate(lin, f_lin):

            x_points = lin
            y_points = f_lin
            spaces_between_Xs = abs(lin[-1] - lin[-2])

            def TDMAsolver(a, b, c, d):

                nf = len(d)  # number of equations
                ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
                for it in range(1, nf):
                    # try:
                    mc = ac[it - 1] / bc[it - 1]
                    # if mc == np.inf:
                    #     mc = 6.982608224074151e+38
                    # if bc[it - 1] < 1e-7:
                    #     print(bc[it - 1])
                    # elif mc < 1e-7 or mc > 10000:
                    #     print(mc)
                    # elif cc[it - 1] < 1e-7 or cc[it - 1] > 10000:
                    #     print(cc[it - 1])
                    # elif dc[it - 1] < 1e-7 or dc[it - 1] > 10000:
                    #     print(dc[it - 1])
                    # if mc == np.inf or mc == np.NINF or mc == np.NAN:
                    #     print(mc)
                    bc[it] = bc[it] - mc * cc[it - 1]
                    # if bc[it] == np.inf:
                    #     bc[it] = 6.982608224074151e+38
                    # dc[it] = dc[it] - mc * dc[it - 1]
                    # if dc[it] == np.inf:
                    #     dc[it] = 6.982608224074151e+38
                    # if bc[it] < 1e-7 or bc[it] > 1e38:
                    #     print(bc[it])
                    dc[it] = dc[it] - mc * dc[it - 1]
                    # if dc[it] < 1e-7 or dc[it] > 1e38:
                    #     print(dc[it])
                    # except:
                    #     print(mc)

                xc = bc
                xc[-1] = dc[-1] / bc[-1]
                # print(dc[-1])
                # print(bc[-1])

                for il in range(nf - 2, -1, -1):
                    xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

                return xc

            third = np.ones(len(x_points) - 2)
            sec = np.insert(np.insert(4 * np.ones(len(x_points) - 3), 0, 2), len(x_points) - 2, 7)
            # print(sec)
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
                # if x == float(b):
                #     k = len(lin) - 1
                # else:
                x -= lin[0]
                k = abs(x / spaces_between_Xs)
                if int(k) == len(lin) - 1:
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
                    root_of_t = intersections(numBezier[0], lambda t: x, 0, 1)
                    # print(root_of_t)
                    return numBezier[1](root_of_t)

                return GetT(x)

            return GetY

        def get_avg_y(f, x, slices):

            if slices != 800:
                a = np.empty(shape=[200000, slices])
                for i in range(0, 200000):
                    a[i] = f(x)
            else:
                a = np.empty(shape=[100000, slices])
                for i in range(0, 100000):
                    a[i] = f(x)

            return np.float32(np.average(a, axis=0))

        def get_avg_y1(f, x, slices):
            if slices != 800:
                a = np.empty(shape=[17000, slices])
                for i in range(0, 17000):
                    for j in range(slices):
                        a[i][j] = f(x[j])
            else:
                a = np.empty(shape=[12000, slices])
                for i in range(0, 12000):
                    for j in range(slices):
                        a[i][j] = f(x[j])

            return np.float32(np.average(a, axis=0))

        tempindication = 0
        try:
            f(np.array([1,2,3,4]))
            tempindication += 1
        except:
            pass
        if tempindication == 1:
            x_p = np.linspace(a, b, 30)
            tempY1 = f(x_p[0])
            tempY2 = f(x_p[-1])
            if abs(tempY2 - tempY1) < 10000:
                slices = 30
                y_p = get_avg_y(f, x_p, slices)
            else:
                slices = 800
                x_p = np.linspace(a, b, 800)
                y_p = get_avg_y(f, x_p, slices)
        else:
            x_p = np.linspace(a, b, 30)
            tempY1 = f(x_p[0])
            tempY2 = f(x_p[-1])
            if abs(tempY2 - tempY1) < 10000:
                slices = 30
                y_p = get_avg_y1(f, x_p, slices)
            else:
                slices = 800
                x_p = np.linspace(a, b, 800)
                y_p = get_avg_y1(f, x_p, slices)
            # y_p = get_avg_y1(f, x_p)
        # def amount_slices(Time, line):
        #     t0 = time.time()
        #     f(a)
        #     ft = time.time() - t0
        #     if ft == 0:
        #         ft = 1e-5
        #     ft1 = int(maxtime // (ft))
        #     if ft1 == 0:
        #         return None
        #     return ft1
        # if tempindication == 1:
        #     x_p = np.linspace(a, b, 30)
        #     amount = amount_slices(maxtime, x_p)
        #     if amount != 0:
        #         slices = amount // 30
        #         print(slices)
        #         y_p = get_avg_y(f, x_p, slices)
        #     else:
        #         return None
        # else:
        #     x_p = np.linspace(a, b, 30)
        #     amount = amount_slices(maxtime, x_p[0])
        #     if amount != 0:
        #         slices = amount // 30
        #         print(slices)
        #         y_p = get_avg_y1(f, x_p, slices)
        #     else:
        #         return None
            # y_p = get_avg_y1(f, x_p)
        interpolation = interpolate(x_p, y_p)
        return interpolation
        # return result



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import random


class TestAssignment4(unittest.TestCase):

    # def test_return(self):
    #     # f = NOISY(0.01)(poly(1,1,1))
    #     f = NOISY(0.01)(lambda x: math.e ** math.e ** x)
    #     ass4 = Assignment4A()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
    #     T = time.time() - T
    #     self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))
    #
    #     ass4 = Assignment4A()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
    #     T = time.time() - T
    #     self.assertGreaterEqual(T, 5)

    # def test_err(self):
    #     f = poly(1,1,1)
    #     nf = NOISY(1)(f)
    #     ass4 = Assignment4A()
    #     T = time.time()
    #     ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
    #     T = time.time() - T
    #     mse=0
    #     for x in np.linspace(0,1,100):
    #         self.assertNotEqual(f(x), nf(x))
    #         mse+= (f(x)-ff(x))**2
    #     mse = mse/100
    #     print(mse)

    def test_err_2(self):
        # f = poly(1, 2, 3, 4, 1, 2, 3, 4)
        f = (lambda x: math.e ** math.e ** x)
        # df = DELAYED(0.01)(f)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=-10, b=1, d=50, maxtime=20)
        # print(ff)
        T = time.time() - T
        # print("done in ", T)
        mse = 0
        uniform = np.random.uniform(low=-10, high=1, size=100)
        for x in uniform:
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 100
        print(mse)






if __name__ == "__main__":
    unittest.main()


# f = poly(1,1,1)
# nf = NOISY(0.01)(f)
# print(nf(1))
# print(nf(1) - f(1))