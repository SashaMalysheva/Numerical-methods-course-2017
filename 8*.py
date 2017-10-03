import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class Spline3:
    def __init__(self, x, y, df0=0, dfn=0):
        self.x = np.copy(x)
        self.a = np.copy(y[:-1])

        n = x.shape[0]
        h = np.zeros(n - 1)
        df = np.zeros(n - 1)
        for i in range(n - 1):
            h[i] = x[i + 1] - x[i]
            df[i] = (y[i + 1] - y[i]) / h[i]

        self.b = np.zeros(n)
        matrix = np.zeros((n, n))
        coeff = np.zeros(n)

        for i in range(1, n - 1):
            matrix[i, i - 1] = 1 / h[i - 1]
            matrix[i, i + 1] = 1 / h[i]
            matrix[i, i] = 2 * (1 / h[i - 1] + 1 / h[i])
            coeff[i] = 3 * (df[i] / h[i] + df[i - 1] / h[i - 1])

        matrix[0, 0] = 1
        coeff[0] = df0

        matrix[n - 1, n - 1] = 1
        coeff[n - 1] = dfn

        self.b = np.linalg.solve(matrix, coeff)

        self.c = np.zeros(n - 1)
        self.d = np.zeros(n - 1)

        for i in range(n - 1):
            self.c[i] = (3 * df[i] - self.b[i + 1] - 2 * self.b[i]) / h[i]
            self.d[i] = (self.b[i] + self.b[i + 1] - 2 * df[i]) / h[i] ** 2

    def get_value(self, x):
        n = self.x.shape[0] - 1
        k = n - 1
        for i in range(n - 1):
            if self.x[i] <= x < self.x[i + 1]:
                k = i
                break
        dx = x - self.x[k]
        return self.a[k] + self.b[k] * dx + self.c[k] * dx ** 2 + self.d[k] * dx ** 3


class Spline2:
    def __init__(self, x, y, df0):
        self.x = np.copy(x)
        self.y = np.copy(y)

        self.a = np.zeros(n - 1)
        self.b = np.zeros(n - 1)
        self.c = np.zeros(n - 1)

    def get_value(self, x):
        n = self.x.shape[0] - 1
        k = n - 1
        for i in range(n - 1):
            if self.x[i] <= x < self.x[i + 1]:
                k = i
                break
        return self.a[k] + self.b[k] * x + self.c[k] * x ** 2


def print_res(x, y, y_new, name):
    average_abs_error_spline = np.average(np.abs(y_new - y))
    max_abs_error_spline = np.max(np.abs(y_new - y))
    average_relative_error_spline = np.average(np.abs(y_new - y) / y)
    max_relative_error_spline = np.max(np.abs(y_new - y) / y)

    print(name + "_average_abs_error = ", average_abs_error_spline)
    print(name + "_max_abs_error = ", max_abs_error_spline)
    print(name + "_average_relative_error = ", average_relative_error_spline)
    print(name + "_max_relative_error = ", max_relative_error_spline)

if __name__ == "__main__":
    with open('10.txt', 'r') as f:
        l = f.read().split()
        n = int(l[0])
        cnts = list(map(int, l[1:4]))
        x = np.array(list(map(float, l[4:4 + n])))
        y = np.array(list(map(float, l[4 + n:])))

        for cnt in cnts + [n - 1]:
            print("cnt = ", cnt)
            x_subset = x[0:n + 1:int(n / cnt)]
            y_subset = y[0:n + 1:int(n / cnt)]

            s = Spline3(x_subset, y_subset)
            y_new1 = np.array([s.get_value(x[i]) for i in range(n)])
            print_res(x, y, y_new1, "natural_spline")
            plt.plot(x, y_new1, label='natural_spline')

            s = Spline2(x_subset, y_subset, 0)
            y_new = np.array([s.get_value(x[i]) for i in range(n)])
            y_new = interpolate.interp1d(x_subset, y_subset)(x)
            print_res(x, y, y_new, "spline2")
            plt.plot(x, y_new, label='spline2')
            plt.plot(x, y, 'r--', label='real')
            plt.plot(x_subset, y_subset, 'ok', label='Known points')
            plt.legend()
            plt.savefig('10_deg=' + str(cnt) + '.png')
            plt.close()
            print("number of points where interpolation is better = ",
                  np.sum(np.abs(y_new1 - y) < np.abs(y_new - y)))

            print("-----------------")

        cnt = 10
        min_error_s2 = 1e10
        best_sample_s2 = np.concatenate((np.random.choice(n - 2, cnt, replace=False) + 1, np.array([0, n - 1])))
        min_error_s3 = 1e10
        best_sample_s3 = np.concatenate((np.random.choice(n - 2, cnt, replace=False) + 1, np.array([0, n - 1])))

        for it in range(1000):
            indexes = np.concatenate((np.random.choice(n - 2, cnt, replace=False) + 1, np.array([0, n - 1])))
            indexes.sort()
            x_subset = x[indexes]
            y_subset = y[indexes]

            s = Spline3(x_subset, y_subset)
            y_new = np.array([s.get_value(x[i]) for i in range(n)])
            average_abs_error_spline = np.average(np.abs(y_new - y))
            if average_abs_error_spline < min_error_s3:
                min_error_s3 = average_abs_error_spline
                best_sample_s3 = indexes

            s = Spline2(x_subset, y_subset, 0)
            y_new = np.array([s.get_value(x[i]) for i in range(n)])
            y_new = interpolate.interp1d(x_subset, y_subset)(x)
            average_abs_error_spline = np.average(np.abs(y_new - y))
            if average_abs_error_spline < min_error_s2:
                min_error_s2 = average_abs_error_spline
                best_sample_s2 = indexes

        x_subset = x[best_sample_s3]
        y_subset = y[best_sample_s3]
        s = Spline3(x_subset, y_subset)
        y_new = np.array([s.get_value(x[i]) for i in range(n)])
        print_res(x, y, y_new, "best_s3")
        plt.plot(x, y_new, label='best_s3')

        x_subset = x[best_sample_s2]
        y_subset = y[best_sample_s2]
        s = Spline2(x_subset, y_subset, 0)
        y_new = np.array([s.get_value(x[i]) for i in range(n)])
        y_new = interpolate.interp1d(x_subset, y_subset)(x)

        print_res(x, y, y_new, "best_s2")
        plt.plot(x, y_new, label='best_s2')
        plt.plot(x, y, 'r--', label='real')
        plt.plot(x_subset, y_subset, 'ok', label='Known points')
        plt.legend()
        plt.savefig('10best' + str(cnt) + '.png')
        plt.close()