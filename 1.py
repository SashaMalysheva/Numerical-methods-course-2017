import numpy as np
import math
import matplotlib.pyplot as plt
import random


def poly_from_roots(L, R, n):
    roots = np.random.random(n) * (R - L) + L
    poly = np.poly(roots)
    return roots, poly

#theorema o upper bound of roots
def theorem(poly):
    m = 0
    for a in poly:
        if a > 0 or m == 0:
            m += 1
            continue
        else:
            break
    if m >= poly.shape[0]:
        return
    A = np.max(np.abs(poly))
    return 1 + (A / poly[0]) ** (1 / m)


def estimate_roots(L, R, n, roots, poly):
    lf = 0
    rf = 0
    altern = np.ones(n + 1)
    for i in range(n - 1, -1, -2):
        altern[i] *= -1
    max_pos = theorem(poly)
    print("max_pos", max_pos)
    min_pos = theorem(poly[::-1] / poly[-1])
    if min_pos is not None:
        min_pos = 1 / min_pos
    min_neg = theorem(poly * altern)
    if min_neg is not None:
        min_neg = -min_neg
    max_neg = theorem((poly * altern)[::-1] / poly[-1])
    if max_neg is not None:
        max_neg = -1 / max_neg

    roots.sort(axis=0)
    if roots[0] > 0:
        lf = roots[0] - min_pos
    else:
        lf = max_neg - roots[0]
    if roots[roots.shape[0] - 1] > 0:
        rf = max_pos - roots[roots.shape[0] - 1]
    else:
        rf = roots[roots.shape[0] - 1] - min_neg
    return lf, rf


def estimate_change_coeff(L, R, n):
    x = []
    y = []
    for j in np.arange(0.999, 1.001, 0.0001):
        x.append(j)
        mean_error = 0
        for i in range(100):
            roots, poly = poly_from_roots(L, R, n)
            old_roots = np.roots(poly)
            poly[1] *= j
            new_roots = np.roots(poly)
            error = abs(abs(old_roots - new_roots) / old_roots)
            mean_error += sum(error)
        mean_error /= 100 * n
        y.append(mean_error)
        print("for coeff error = ", j, "mean_error = ", mean_error)
    plt.plot(x, y)
    plt.title(str(L) + "_" + str(R))
    plt.savefig("plot_change_coeff" + str(L) + "_" + str(R) + ".png")
    plt.close()


def estimate_round_coeff(L, R, n):
    x = []
    y = []
    for k in range(2, 6):
        mean_error = 0
        for i in range(500):
            roots, poly = poly_from_roots(L, R, n)
            old_roots = np.roots(poly)
            new_poly = np.floor(poly * (10 ** k)) * (10 ** -k)
            new_roots = np.roots(new_poly)
            error = abs(abs(old_roots - new_roots) / old_roots)
            mean_error += sum(error)
        mean_error /= 100 * n
        print("for k = ", k, "mean_error = ", mean_error)
        x.append(k)
        y.append(mean_error)
    plt.plot(x, y)
    plt.title(str(L) + "_" + str(R))
    plt.savefig("plot_round_coeff" + str(L) + "_" + str(R) + ".png")
    plt.close()


def find_root(l, r, poly):
    der = poly.deriv()
    der2 = der.deriv()
    x0 = random.uniform(l, r)
    while der2(x0) * poly(x0) <= 0:
        x0 = random.uniform(l, r)

    cur_x = x0
    for i in range(250):
        t1 = poly(cur_x)
        t2 = der(x0)
        cur_x = cur_x - poly(cur_x) / der(x0)
    return cur_x


def estimate_newton(L, R, n):
    #    for i in range(100):
    roots, poly = poly_from_roots(L, R, n)
    poly1d = np.poly1d(poly)
    roots.sort(axis=0)
    for i in range(roots.shape[0]):
        print("root = ", roots[i])
        if i == 0:
            new_root = find_root(roots[0] - 1, (roots[0] + roots[1]) / 2, poly1d)
        elif i == roots.shape[0] - 1:
            new_root = find_root((roots[i - 1] + roots[i]) / 2, roots[i] + 1, poly1d)
        else:
            new_root = find_root((roots[i - 1] + roots[i]) / 2, (roots[i] + roots[i + 1]) / 2, poly1d)
        diff = roots[i] - new_root
        print("absolute error = ", abs(diff))


def do_all_tests(L, R, n):
    print("bounds = ", L, R)
    estimate_newton(L, R, n)


def find_root_transc():
    L = -math.pi / 3
    R = math.pi / 3
    x0 = random.uniform(L, R)
    denom = 1 - math.sin(x0)
    cur_x = x0
    for i in range(200):
        cur_x -= (cur_x + math.cos(cur_x) - 1) / denom
    print(cur_x)


if __name__ == "__main__":
    n = 10
    L = -100
    R = 100
    do_all_tests(L, R, n)
    L = 0
    R = 100
    do_all_tests(L, R, n)
    L = -1
    R = 1
    do_all_tests(L, R, n)
    L = 0
    R = 1
    do_all_tests(L, R, n)
    find_root_transc()
