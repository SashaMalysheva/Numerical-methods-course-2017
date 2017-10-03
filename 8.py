import numpy as np
import matplotlib.pyplot as plt


def find_dy(x, y):
    n = x.shape[0]
    h = np.zeros(n - 1)
    d = np.zeros(n - 1)
    dy = np.zeros(n)
    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]
        d[i] = (y[i + 1] - y[i]) / h[i]

    for i in range(1, n - 1):
        dy[i] = (d[i] + d[i - 1]) / 2

    dy[0] = d[0]
    dy[n - 1] = d[n - 2]

    return dy


def HermitSpline(x, y, dy, x0):
    if dy is None:
        dy = find_dy(x, y)
    a = np.copy(y[:-1])
    b = np.copy(dy)
    n = x.shape[0]
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(n - 1):
        c[i] = (3 * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - dy[i + 1] - 2 * dy[i]) / (x[i + 1] - x[i])
        d[i] = (dy[i] + dy[i + 1] - 2 * (y[i + 1] - y[i]) / (x[i + 1] - x[i])) / (x[i + 1] - x[i]) ** 2

    n = x.shape[0] - 1
    k = n - 1
    for i in range(n - 1):
        if x[i] <= x0 < x[i + 1]:
            k = i
            break
    dx = x0 - x[k]
    return a[k] + b[k] * dx + c[k] * dx ** 2 + d[k] * dx ** 3


class HermitPoly:
    def __init__(self, x, y, dy=None):
        if dy is None:
            dy = find_dy(x, y)

        self.x = np.copy(x)
        self.a = np.copy(y[:-1])
        self.b = np.copy(dy[:-1])

        n = x.shape[0]
        h = np.zeros(n - 1)
        df = np.zeros(n - 1)
        for i in range(n - 1):
            h[i] = x[i + 1] - x[i]
            df[i] = (y[i + 1] - y[i]) / h[i]

        self.c = np.zeros(n)
        self.d = np.zeros(n)

        for i in range(n - 1):
            self.c[i] = (3 * df[i] - dy[i + 1] - 2 * self.b[i]) / h[i]
            self.d[i] = (dy[i] + dy[i + 1] - 2 * df[i]) / h[i] ** 2

    def get_value(self, x):
        n = self.x.shape[0] - 1
        k = n - 1
        for i in range(n - 1):
            if self.x[i] <= x < self.x[i + 1]:
                k = i
                break
        dx = x - self.x[k]
        return self.a[k] + self.b[k] * dx + self.c[k] * dx ** 2 + self.d[k] * dx ** 3


def print_res(y, y_new, name):
    average_abs_error_spline = np.average(np.abs(y_new - y))
    max_abs_error_spline = np.max(np.abs(y_new - y))
    average_relative_error_spline = np.average(np.abs(y_new - y) / y)
    max_relative_error_spline = np.max(np.abs(y_new - y) / y)
    print(name + "_average_abs_error = ", average_abs_error_spline)
    print(name + "_max_abs_error = ", max_abs_error_spline)
    print(name + "_average_relative_error = ", average_relative_error_spline)
    print(name + "_max_relative_error = ", max_relative_error_spline)


if __name__ == "__main__":
    with open('5.txt', 'r') as f:
        l = f.read().split()
        n = int(l[0])
        cnts = list(map(int, l[1:4]))
        x = np.array(list(map(float, l[4:4 + n])))
        y = np.array(list(map(float, l[4 + n: 4 + 2 * n])))
        dy = np.array(list(map(float, l[4 + 2 * n:4 + 3 * n])))

        for cnt in cnts + [n - 1]:
            print("cnt = ", cnt)
            x_subset = x[0:n + 1:int(n / cnt)]
            y_subset = y[0:n + 1:int(n / cnt)]
            dy_subset = dy[0:n + 1:int(n / cnt)]
            if x[-1] != x_subset[-1]:
                x_subset = np.append(x_subset, x[-1])
                y_subset = np.append(y_subset, y[-1])
                dy_subset = np.append(dy_subset, dy[-1])
            # print(x_subset)

            y_spline = np.array([HermitSpline(x_subset, y_subset, dy_subset, x[i]) for i in range(n)])
            print_res(y, y_spline, "hermit_spline")
            plt.plot(x, y_spline, label='hermit_spline')

            y_new = np.array([HermitSpline(x_subset, y_subset, None, x[i]) for i in range(n)])
            print_res(y, y_new, "hermit_spline_without_der")
            plt.plot(x, y_new, label='hermit_spline_without_der')

            hermit_poly = HermitPoly(x_subset, y_subset, dy_subset)
            y_poly = np.array([hermit_poly.get_value(x[i]) for i in range(n)])
            print_res(y, y_poly, "hermit_poly")
            plt.plot(x, y_poly, label='hermit_poly')

            print("spline is better in ", np.sum(np.abs(y_spline - y) <= np.abs(y_poly - y)), "points")

            print("dy_error = ", find_dy(x_subset, y_subset) - dy_subset)
            dy_spline = find_dy(x, y_spline)
            print_res(dy, dy_spline, "dy_spline")
            dy_poly = find_dy(x, y_poly)
            print_res(dy, dy_poly, "dy_poly")

            plt.plot(x, y, 'r--', label='real')
            plt.plot(x_subset[1:-1], y_subset[1:-1], 'ok', label='Known points')
            plt.legend()
            plt.savefig('8_deg=' + str(cnt) + '.png')
            plt.close()

        cnt = 15
        min_error_poly = 1e10
        best_sample_poly = np.concatenate((np.random.choice(n - 2, cnt, replace=False) + 1, np.array([0, n - 1])))
        min_error_spline = 1e10
        best_sample_spline = np.concatenate((np.random.choice(n - 2, cnt, replace=False) + 1, np.array([0, n - 1])))

        for it in range(100):
            indexes = np.concatenate((np.random.choice(n - 2, cnt, replace=False) + 1, np.array([0, n - 1])))
            indexes.sort()
            x_subset = x[indexes]
            y_subset = y[indexes]
            dy_subset = dy[indexes]

            y_new = np.array([HermitSpline(x_subset, y_subset, dy_subset, x[i]) for i in range(n)])
            average_abs_error_spline = np.average(np.abs(y_new - y))
            if average_abs_error_spline < min_error_spline:
                min_error_spline = average_abs_error_spline
                best_sample_spline = indexes

            hermit_poly = HermitPoly(x_subset, y_subset, dy_subset)
            y_poly = np.array([hermit_poly.get_value(x[i]) for i in range(n)])
            average_abs_error_spline = np.average(np.abs(y_new - y))
            if average_abs_error_spline < min_error_poly:
                min_error_poly = average_abs_error_spline
                best_sample_poly = indexes

        x_subset = x[best_sample_spline]
        y_subset = y[best_sample_spline]
        dy_subset = dy[best_sample_spline]
        y_new = np.array([HermitSpline(x_subset, y_subset, dy_subset, x[i]) for i in range(n)])
        print_res(y, y_new, "best_spline")
        plt.plot(x, y_new, label='best_spline')

        x_subset = x[best_sample_poly]
        y_subset = y[best_sample_poly]
        hermit_poly = HermitPoly(x_subset, y_subset, dy_subset)
        y_poly = np.array([hermit_poly.get_value(x[i]) for i in range(n)])
        print_res(y, y_new, "best_poly")
        plt.plot(x, y_new, label='best_poly')
        plt.plot(x, y, 'r--', label='real')
        plt.plot(x_subset[1:-1], y_subset[1:-1], 'ok', label='Known points')
        plt.legend()
        plt.savefig('8best.png')
        plt.close()
        print("best_sample_spline", best_sample_spline)
        print("best_sample_poly", best_sample_poly)