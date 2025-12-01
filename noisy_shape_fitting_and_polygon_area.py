"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import math
import time
import random
from functionUtils import AbstractShape


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        arr_pts_on_figure = contour(1800)
        def polygonAREA(x, y):
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        #     area = 0
        #     j = npoints - 1
        #     for i in range((npoints)):
        #         area += (x[j] + x[i]) * (y[j] - y[i])
        #         j = i
        #     return area / 2
        # return np.float32(polygonAREA(arr_pts_on_figure[:, 0], arr_pts_on_figure[:, 1], len(arr_pts_on_figure)))
        return np.float32(polygonAREA(arr_pts_on_figure[:, 0], arr_pts_on_figure[:, 1]))


    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        arr_div_points = np.array([sample(), ])
        x_min = arr_div_points[0][0]
        x_max = arr_div_points[0][0]
        for i in range(1, 50000):
            arr_div_points = np.append(arr_div_points, [np.array(sample())], axis=0)
            if x_min > arr_div_points[i][0]:
                x_min = arr_div_points[i][0]
            elif x_max < arr_div_points[i][0]:
                x_max = arr_div_points[i][0]


        center_shape = np.average(arr_div_points, axis=0)
        origin = center_shape
        refvec = [0, 1]

        def clockwiseangle_and_distance(point):
            vector = [point[0] - origin[0], point[1] - origin[1]]
            lenvector = math.hypot(vector[0], vector[1])
            if lenvector == 0:
                return -math.pi, 0
            normalized = [vector[0] / lenvector, vector[1] / lenvector]
            dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
            diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]
            angle = math.atan2(diffprod, dotprod)
            if angle < 0:
                return 2 * math.pi + angle, lenvector
            return angle, lenvector
        temp_indication = 0
        arr_div_points = np.array(sorted(arr_div_points, key=clockwiseangle_and_distance))
        arr_div_points_otherVERSION = arr_div_points.copy()
        if abs(x_max - x_min) < 25:
            arr_div_points = np.split(arr_div_points, 10)
        elif abs(x_max - x_min) < 50:
            arr_div_points = np.split(arr_div_points, 50)
        elif abs(x_max - x_min) < 100:
            arr_div_points = np.split(arr_div_points, 100)
        elif abs(x_max - x_min) < 10000:
            arr_div_points = np.split(arr_div_points, 150)
        elif abs(x_max - x_min) >= 10000:
            temp_indication = 1
        if temp_indication != 1:
            arr_pts_on_figure = np.empty(shape=[1, 2])
            arr_pts_on_figure[0] = np.average(arr_div_points[0], axis=0)
            for i in arr_div_points[1:]:
                arr_pts_on_figure = np.append(arr_pts_on_figure, [np.average(i, axis=0)], axis=0)
        if temp_indication == 1:
            origin = [2, 3]
            arr_div_points_otherVERSION = np.array(sorted(arr_div_points_otherVERSION, key=clockwiseangle_and_distance))
        if temp_indication != 1:
            class MyShape(AbstractShape):
                def __init__(self, arr_pts_on_figure):
                    self.arr_pts_on_figure = arr_pts_on_figure

                def area(self):
                    def polygonAREA(x, y):
                        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    return polygonAREA(arr_pts_on_figure[:, 0], arr_pts_on_figure[:, 1])

                def contour(self, n: int):
                    pass

                def sample(self, n: int):
                    pass
        elif temp_indication == 1:
            class MyShape(AbstractShape):
                def __init__(self, arr_div_points_otherVERSION):
                    self.arr_div_points_otherVERSION = arr_div_points_otherVERSION

                def area(self):
                    def polygonAREA1(x, y):
                        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

                    return polygonAREA1(arr_div_points_otherVERSION[:, 0], arr_div_points_otherVERSION[:, 1])

                def contour(self, n: int):
                    pass

                def sample(self, n: int):
                    pass
        if temp_indication != 1:
            return MyShape(arr_pts_on_figure)
        elif temp_indication == 1:
            return MyShape(arr_div_points_otherVERSION)



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #
    #     def sample():
    #         time.sleep(7)
    #         return circ()
    #
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=sample, maxtime=5)
    #     T = time.time() - T
    #     self.assertTrue(isinstance(shape, AbstractShape))
    #     self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
