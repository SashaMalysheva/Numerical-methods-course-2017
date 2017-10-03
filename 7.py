import numpy as np
import matplotlib.pyplot as plt


def lagrange_interpolation(x, y):
    n = len(x)
    res = np.poly1d([0])

    for i in range(n):
        q = np.poly1d([1])
        for j in range(n):
            if j != i:
                q = q * (np.poly1d([1, -x[j]]) / np.poly1d([x[i] - x[j]]))[0]
        res = res + q * np.poly1d([y[i]])

    return res


def ordinary_least_squares(x, y, deg):
    x_powers = np.array([np.sum(x ** i) for i in range(2 * deg + 1)])

    matrix = [[x_powers[i + j] for i in range(deg + 1)] for j in range(deg + 1)]
    b = np.array([np.sum(x ** i * y) for i in range(deg + 1)])

    coefficients = np.linalg.solve(matrix, b)

    return np.poly1d(coefficients[::-1])

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
        degrees = list(map(int, l[1:4]))
        x = np.array(list(map(float, l[4:4 + n])))
        y = np.array(list(map(float, l[4 + n:])))

        for deg in degrees:
            print("deg = ", deg)
            x_for_interpolation = x[0:n + 1:int(n / deg)]
            y_for_interpolation = y[0:n + 1:int(n / deg)]

            poly_interpolation = lagrange_interpolation(x_for_interpolation, y_for_interpolation)
            y_interpolated = np.array([poly_interpolation(x[i]) for i in range(n)])
            y_new = np.copy(y_interpolated)
            print_res(x, y, y_new, "interpolation")
            plt.plot(x, y_new, label='interpolation')

            poly_ols = ordinary_least_squares(x, y, deg)
            y_ols = np.array([poly_ols(x[i]) for i in range(n)])
            y_new = np.copy(y_ols)

            print_res(x, y, y_new, "ols")
            plt.plot(x, y_new, label='ols')

            plt.plot(x, y, '--k', label='True values')
            plt.plot(x_for_interpolation, y_for_interpolation, 'ok', label='Known points')
            plt.legend()
            plt.savefig('7deg=' + str(deg) + '.png')
            plt.close()

            print("number of points where interpolation is better = ",
                  np.sum(np.abs(y_interpolated - y) < np.abs(y_ols - y)))

            print("-----------------")

    # find best interpolation polynomial
    min_average_relative_error = 100
    best_deg = 0
    y_interpolated_best = None

    for deg in range(1, int(n / 2)):
        x_for_interpolation = x[0:n + 1:int(n / deg)]
        y_for_interpolation = y[0:n + 1:int(n / deg)]

        poly_interpolation = lagrange_interpolation(x_for_interpolation, y_for_interpolation)
        y_interpolated = np.array([poly_interpolation(x[i]) for i in range(n)])
        average_relative_error = np.average(np.abs(y_interpolated - y) / y)
        if average_relative_error < min_average_relative_error:
            min_average_relative_error = average_relative_error
            best_deg = deg
            y_interpolated_best = y_interpolated

    for deg in range(1, int(n / 2)):
        x_for_interpolation = x[0:n + 1:int(n / deg)]
        y_for_interpolation = y[0:n + 1:int(n / deg)]
        poly_ols = ordinary_least_squares(x, y, deg)
        y_ols = np.array([poly_ols(x[i]) for i in range(n)])
        average_relative_error_ols = np.average(np.abs(y_ols - y) / y)
        if average_relative_error_ols > min_average_relative_error:
            print("bigger - deg: ", deg, " average_relative_error_ols: ", average_relative_error_ols)
        if average_relative_error_ols < min_average_relative_error:
            print("less - deg: ", deg, " average_relative_error_ols: ", average_relative_error_ols)

    print("best deg for interpolation = ", best_deg)
    print("average_relative_error_interpolation = ", min_average_relative_error)

    poly_ols = ordinary_least_squares(x, y, best_deg)
    y_ols = np.array([poly_ols(x[i]) for i in range(n)])
    average_relative_error_ols = np.average(np.abs(y_ols - y) / y)
    print("average_relative_error_ols = ", average_relative_error_ols)
