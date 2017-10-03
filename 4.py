import numpy as np
import math

eps = 1e-8


def create_random_matrix(n, l=0, r=1):
    return np.random.rand(n, n) * (r - l) + l


def create_random_vector(n, l = 0, r = 1):
    return np.random.random(n) * (r - l) + l


def round(a, n):
    if n == -1:
        return a
    else:
        return (np.rint(a * 10**n))/10**n


def QR(A, b, round_n):
    cnt_iter = 0
    n = b.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            tan = -A[j, i]/A[i, i]
            cnt_iter += 1
            cos = 1/math.sqrt(1 + tan**2)
            cnt_iter += 4
            sin = tan*cos
            cnt_iter += 1

            r1 = A[i, :].copy()
            r2 = A[j, :].copy()

            A[j, :] = sin * r1 + cos * r2
            cnt_iter += 2*n
            A[i, :] = -sin * r2 + cos * r1
            cnt_iter += 2*n

            r1 = b[i]
            r2 = b[j]
            b[i] = cos*r1 - sin*r2
            b[j] = sin*r1 + cos*r2
            cnt_iter += 4

            A = round(A, round_n)
            b = round(b, round_n)
    return A, b, cnt_iter

if __name__ == "__main__":
        n = 10
        A = create_random_matrix(n)
        x = create_random_vector(n)
        b = np.dot(A, x)

        print(A)
        print(b)
        A, b, cnt_iter = QR(A, b, 6)
        print("cnt_iter = ", cnt_iter)
        print(A)
        print(b)
        print(A.dot(x))
        ans = [0]*n
        for i in range(n - 1, -1, -1):
            ans[i] = b[i]
            for j in range(i + 1, n):
                ans[i] -= ans[j]*A[i][j]
            ans[i] /= A[i][i]

        print(ans)
        print(np.sum(ans - x))
