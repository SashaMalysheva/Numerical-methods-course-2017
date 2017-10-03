import numpy as np
import numpy.linalg as la


def are_submatrices_not_singular(matrix):
    for i in range(matrix.shape[0]):
        if la.det(matrix[:i + 1, :i + 1]) == 0:
            return False
    return True


def find_matrix_with_cond(l, r):
    while True:
        matrix = np.random.randint(-10, 10, size=(n, n))
        if l <= la.cond(matrix) <= r and are_submatrices_not_singular(matrix):
            return matrix


def calc_lu(matrix, perm):
    l = np.zeros(matrix.shape)
    u = np.zeros(matrix.shape)
    cnt = 0

    for m in range(matrix.shape[0]):
        for i in range(m):
            u[i][m] = matrix[perm[i]][m] - np.dot(l[i, :i], u[:i, m])
            cnt += i
        l[m][m] = 1

        for i in range(m):
            l[m][i] = (matrix[perm[m]][i] - np.dot(l[m, :i], u[:i, i])) / u[i][i]
            cnt += i + 1

        u[m][m] = matrix[perm[m]][m] - np.dot(l[m, :m], u[:m, m])
        cnt += m

        if abs(u[m][m]) < __eps:
            print("Swap")

    return l, u, cnt


def calc_lu_max(matrix, perm):
    l = np.zeros(matrix.shape)
    u = np.zeros(matrix.shape)
    cnt = 0

    for m in range(matrix.shape[0]):
        i_max = m
        j_max = m
        for i in range(m, n):
            for j in range(m, n):
                if matrix[i, j] > matrix[i_max, j_max]:
                    i_max = i
                    j_max = j
        matrix[m, m], matrix[i_max, j_max] = matrix[i_max, j_max], matrix[m, m]
        cnt += (n - m) ** 2 + 1
        for i in range(m):
            u[i][m] = matrix[perm[i], m] - np.dot(l[i, :i], u[:i, m])
            cnt += i
        l[m][m] = 1

        for i in range(m):
            l[m][i] = (matrix[perm[m], i] - np.dot(l[m, :i], u[:i, i])) / u[i, i]
            cnt += i + 1

        u[m][m] = matrix[perm[m], m] - np.dot(l[m, :m], u[:m, m])
        cnt += m

        if abs(u[m, m]) < __eps:
            print("Swap")

    return l, u, cnt


if __name__ == "__main__":
    n = 10
    min_cond = 10000
    max_cond = 11000
    __eps = 1e-7

    matrix = find_matrix_with_cond(min_cond, max_cond)
    print("A = ", matrix)
    print("cond(A)=", la.cond(matrix))

    perm = np.arange(n)
    l, u, cnt = calc_lu(matrix, perm)
    print("l = ", l)
    print("u = ", u)
    print("cnt = ", cnt)

    print("lu = ", np.dot(l, u))
    print(la.norm(np.dot(l, u) - matrix))

    perm = np.arange(n)
    l, u, cnt = calc_lu_max(matrix, perm)
    print("l = ", l)
    print("u = ", u)
    print("cnt !!!!!  = ", cnt)

    print("lu = ", np.dot(l, u))

    x = np.zeros(n) + 1
    print("x = ", x)
    b = np.dot(matrix, x)
    print("b = ", b)

    print('x1')
    x1 = np.linalg.solve(np.dot(l, u), b)
    print(x- x1)
