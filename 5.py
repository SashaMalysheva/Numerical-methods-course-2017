import numpy as np


def create_random_matrix(n, l=0, r=1):
    m = np.random.rand(n, n) * (r - l) + l
    for i in range(n):
        m[i][i] += m[i].sum()
    return m


def create_random_vector(n, l=0, r=1):
    return np.random.random(n) * (r - l) + l


def dist2(a):
    return (a ** 2).sum()


def find_solve_simple(C, g, cur_x, prec, cnt=0):
    new_x = C.dot(cur_x) + g
    print("cur ", cur_x)
    print("new ", new_x)
    print(dist2(new_x - cur_x))
    if dist2(new_x - cur_x) < prec:
        return new_x, 1
    else:
        ans, it = find_solve_simple(C, g, new_x, prec, cnt + 1)
        return ans, it + 1


def find_solve_zeidel1(C, g, cur_x, prec, cnt=0):
    old_x = cur_x.copy()
    for i in range(cur_x.shape[0]):
        cur_x[i] = np.sum(C[i] * cur_x) + g[i]
    print("old ", old_x)
    print("cur ", cur_x)
    print(dist2(old_x - cur_x))
    if dist2(old_x - cur_x) < prec:
        return cur_x, 1
    else:
        ans, it = find_solve_zeidel1(C, g, cur_x, prec, cnt + 1)
        return ans, it + 1


def find_solve_zeidel2(C, g, cur_x, prec, cnt=0):
    old_x = cur_x.copy()
    for i in range(cur_x.shape[0] - 1, -1, -1):
        cur_x[i] = np.sum(C[i] * cur_x) + g[i]
    print("old ", old_x)
    print("cur ", cur_x)
    print(dist2(old_x - cur_x))
    if dist2(old_x - cur_x) < prec:
        return cur_x, 1
    else:
        ans, it = find_solve_zeidel2(C, g, cur_x, prec, cnt + 1)
        return ans, it + 1


def find_solve_zeidel3(C, g, cur_x, prec, cnt=0):
    old_x = cur_x.copy()
    for i in range(0, cur_x.shape[0], 2):
        cur_x[i] = np.sum(C[i] * cur_x) + g[i]
    for i in range(1, cur_x.shape[0], 2):
        cur_x[i] = np.sum(C[i] * cur_x) + g[i]
    print("old ", old_x)
    print("cur ", cur_x)
    print(dist2(old_x - cur_x))
    if dist2(old_x - cur_x) < prec:
        return cur_x, 1
    else:
        ans, it = find_solve_zeidel3(C, g, cur_x, prec, cnt + 1)
        return ans, it + 1


def find_solve_zeidel4(C, g, cur_x, prec, cnt=0):
    old_x = cur_x.copy()
    for i in range(cur_x.shape[0] - 1, -1, -2):
        cur_x[i] = np.sum(C[i] * cur_x) + g[i]
    for i in range(cur_x.shape[0] - 2, -1, -2):
        cur_x[i] = np.sum(C[i] * cur_x) + g[i]
    print("old ", old_x)
    print("cur ", cur_x)
    print(dist2(old_x - cur_x))
    if dist2(old_x - cur_x) < prec:
        return cur_x, 1
    else:
        ans, it = find_solve_zeidel4(C, g, cur_x, prec, cnt + 1)
        return ans, it + 1


if __name__ == "__main__":
    n = 50
    A = create_random_matrix(n)
    x = create_random_vector(n)
    b = np.dot(A, x)

    print(A)
    print(b)

    tau = None
    for i in range(n):
        if tau is None:
            tau = A[i, i]
        elif abs(tau) < abs(A[i, i]):
            tau = A[i, i]

    B = np.eye(n)
    alpha = 1.0 / tau
    C = np.eye(n) - alpha * A
    print("eig ", np.linalg.eig(C))
    g = alpha * b

    print(C)
    print(g)

    x1, cnt_iter1 = find_solve_simple(C, g, np.zeros(n), 0.000001)
    x2, cnt_iter2 = find_solve_zeidel1(C, g, np.zeros(n), 0.000001)
    x3, cnt_iter3 = find_solve_zeidel2(C, g, np.zeros(n), 0.000001)
    x4, cnt_iter4 = find_solve_zeidel3(C, g, np.zeros(n), 0.000001)
    x5, cnt_iter5 = find_solve_zeidel4(C, g, np.zeros(n), 0.000001)

    print("simple: ", np.sum(x1 - x), "iters: ", cnt_iter1)
    print("zeidel 0..N: ", np.sum(x2 - x), "iters: ", cnt_iter2)
    print("zeidel N..0: ", np.sum(x3 - x), "iters: ", cnt_iter3)
    print("zeidel 02..N, 13..N: ", np.sum(x4 - x), "iters: ", cnt_iter4)
    print("zeidel N..20, N..31: ", np.sum(x5 - x), "iters: ", cnt_iter5)
