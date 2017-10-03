import numpy as np
import matplotlib.pyplot as plt


class Spline3:
    def __init__(self, x, y, df0=1, dfn=1):
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


class Spline3Der2:
    def __init__(self, x, y, d2f0=1, d2fn=1):
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

        matrix[0, 0] = 2
        matrix[0, 1] = 1
        coeff[0] = 3 * df[0] - h[0] / (2 * d2f0)

        matrix[n - 1, n - 2] = 2
        matrix[n - 1, n - 1] = 1
        coeff[n - 1] = 3 * df[n - 2] - h[n - 2] / (2 * d2fn)

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


class Spline3NotAKnot:
    def __init__(self, x, y):
        x_lost = x[1]
        y_lost = y[1]

        n = x.shape[0]

        x_new = np.zeros(n - 1)
        y_new = np.zeros(n - 1)

        x_new[0] = x[0]
        y_new[0] = y[0]

        for i in range(2, n):
            x_new[i - 1] = x[i]
            y_new[i - 1] = y[i]

        self.x = x_new
        self.a = y_new

        n = x_new.shape[0]
        h = np.zeros(n - 1)
        df = np.zeros(n - 1)
        for i in range(n - 1):
            h[i] = self.x[i + 1] - self.x[i]
            df[i] = (self.a[i + 1] - self.a[i]) / h[i]

        self.b = np.zeros(n)
        matrix = np.zeros((n - 1, n - 1))
        coeff = np.zeros(n - 1)

        theta0 = x_lost - self.x[0] - 2 * ((x_lost - self.x[0]) ** 2) / (h[0]) + ((x_lost - self.x[0]) ** 3) / (
            h[0] ** 2)
        theta1 = - ((x_lost - self.x[0]) ** 2) / (h[0]) + ((x_lost - self.x[0]) ** 3) / (h[0] ** 2)

        delta0 = y_lost - self.a[0] - 3 * df[0] * ((x_lost - self.x[0]) ** 2) / h[0] + \
                 2 * df[0] * ((x_lost - self.x[0]) ** 3) / (h[0] ** 2)

        matrix[0, 0] = theta0
        matrix[0, 1] = theta1
        coeff[0] = delta0
        for i in range(1, n - 1):
            if i > 1:
                matrix[i, i - 1] = 1 / h[i - 1]
            if i < n - 3:
                matrix[i, i + 1] = 1 / h[i]
            matrix[i, i] = 2 * (1 / h[i - 1] + 1 / h[i])
            coeff[i] = 3 * (df[i] / h[i] + df[i - 1] / h[i - 1])

        self.b[0:n - 1] = np.linalg.solve(matrix, coeff)

        self.c = np.zeros(n - 1)
        self.d = np.zeros(n - 1)

        for i in range(n - 1):
            self.c[i] = (3 * df[i] - self.b[i + 1] - 2 * self.b[i]) / h[i]
            self.d[i] = (self.b[i] + self.b[i + 1] - 2 * df[i]) / h[i] ** 2

    def get_value(self, x):
        k = self.x.shape[0] - 2
        for i in range(self.x.shape[0] - 2):
            if self.x[i] <= x < self.x[i + 1]:
                k = i
                break
        dx = x - self.x[k]
        return self.a[k] + self.b[k] * dx + self.c[k] * dx ** 2 + self.d[k] * dx ** 3


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
    with open('05.txt', 'r') as f:
        l = f.read().split()
        n = int(l[0])
        cnts = list(map(int, l[1:4]))
        x = np.array(list(map(float, l[4:4 + n])))
        y = np.array(list(map(float, l[4 + n:])))

        for cnt in cnts + [n - 1]:
            print("cnt = ", cnt)
            x_subset = x[0:n + 1:int(n / cnt)]
            y_subset = y[0:n + 1:int(n / cnt)]

            s = Spline3(x_subset, y_subset, 0, 0)
            y_new = np.array([s.get_value(x[i]) for i in range(n)])

            print_res(x, y, y_new,"natural_spline")
            plt.plot(x, y_new, label='natural_spline')

            s = Spline3(x_subset, y_subset, 1, 1)
            y_new = np.array([s.get_value(x[i]) for i in range(n)])

            print_res(x, y, y_new, "spline_with_der")
            plt.plot(x, y_new, label='spline_with_der')

            s = Spline3Der2(x_subset, y_subset, 1, 1)
            y_new = np.array([s.get_value(x[i]) for i in range(n)])

            plt.plot(x, y_new, label='spline_2der')
            print_res(x, y, y_new, "spline_2der")

            s = Spline3NotAKnot(x_subset, y_subset)
            y_new = np.array([s.get_value(x[i]) for i in range(n)])

            print_res(x, y, y_new,"spline_not_a_knot")
            plt.plot(x, y_new, label='spline_not_a_knot')
            plt.plot(x, y, 'r--', label='real')
            plt.plot(x_subset, y_subset, 'ok', label='Known points')
            plt.legend()
            plt.savefig('9deg=' + str(cnt) + '.png')
            plt.close()
