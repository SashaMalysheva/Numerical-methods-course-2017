import numpy as np
from random import random


def find_non_singular_matrix():
    while True:
        matrix = np.random.rand(n, n)
        if np.linalg.det(matrix) != 0:
            return matrix


def find_matrix_with_cond(cond):
    rand_matrix = find_non_singular_matrix()
    matrix = np.zeros((n, n))
    max_eig = 1
    min_eig = max_eig / cond
    for i in range(n):
        matrix[i, i] = min_eig + (max_eig - min_eig) * random()

    matrix[0, 0] = min_eig
    matrix[n - 1, n - 1] = max_eig
    print("cond = ", np.linalg.cond(matrix))
    return np.dot(np.dot(np.linalg.inv(rand_matrix), matrix), rand_matrix)


def add_error(b, delta):
    b1 = b * np.random.random(b.shape)
    norm = np.linalg.norm(b - b1) / np.linalg.norm(b)
    new_diff = (b - b1) * delta / norm
    return b - new_diff


n = 5
min_cond = 10000
matrix = find_matrix_with_cond(min_cond)

eig = np.linalg.eig(matrix)
print("eig = ", eig)

matrix = find_matrix_with_cond(min_cond)
print("A = ", matrix)

eig = np.linalg.eig(matrix)
print("eig = ", eig)

cond = np.linalg.cond(matrix)
print("cond(A) = ", cond)

# x = np.linalg.solve(matrix, b)
x = np.zeros(n) + 1
print("x = ", x)
b = np.dot(matrix, x)
print("b = ", b)
delta = 0.01
# b1 = add_error(b, delta)
b1 = np.copy(b)

for i in range(n):
    print("i = ", i)
    b1[i] = b[i] * 1.01
    print("b1 = ", b1)
    db = np.linalg.norm(b - b1) / np.linalg.norm(b)
    print("db = ", db)
    x1 = np.linalg.solve(matrix, b1)
    print(x1)
    dx = np.linalg.norm(x - x1) / np.linalg.norm(x)
    print(dx / db, " <= ", cond)
    b1[i] = b[i]

matrix1 = add_error(matrix, delta)
print("A1 = ", matrix1)
d_matrix = np.linalg.norm(matrix - matrix1) / np.linalg.norm(matrix)
x1 = np.linalg.solve(matrix1, b)
dx = np.linalg.norm(x - x1) / np.linalg.norm(x1)
print(dx / d_matrix, " <= ", cond)
