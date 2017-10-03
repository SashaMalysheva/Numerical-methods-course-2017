import numpy as np
import random


def find_grad_simple(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0],))
    r = A.dot(x_new) - b
    while np.sum(r**2) > prec:
        cnt_op += 1
        x_old = x_new
        r = A.dot(x_old) - b
        alpha = r.dot(r) / r.dot(A.dot(r))
        x_new = x_old - alpha * r
        print("alpha: ", alpha, " r: ", np.sum(r), " intr: ", cnt_op)
    return x_new, cnt_op


def find_grad_smart(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0],))
    r = A.dot(x_new) - b
    while np.sum(r**2) > prec:
        cnt_op += 1
        x_old = x_new
        cur_A = A.dot(r)
        alpha = r.dot(r) / r.dot(cur_A)
        x_new = x_old - alpha * r
        r = r - alpha*cur_A
        print("alpha: ", alpha, " r: ", np.sum(r), " intr: ", cnt_op)
    return x_new, cnt_op


def free_grad(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0], ))
    r = A.dot(x_new) - b
    p = np.zeros((A.shape[0],))
    while np.sum(r**2) > prec:
        j = random.randrange(A.shape[0])
        p[j] = 1
        cnt_op += 1
        x_old = x_new
        r = A.dot(x_old) - b
        alpha = r.dot(p) / p.dot(A.dot(p))
        x_new = x_old - alpha * p
        p[j] = 0
        print("alpha: ", alpha, " r: ", np.sum(r), " intr: ", cnt_op)
    return x_new, cnt_op


def free_grad_smart(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0], ))
    r = A.dot(x_new) - b
    p = np.zeros((A.shape[0],))
    while np.sum(r**2) > prec:
        j = random.randrange(A.shape[0])
        p[j] = 1
        cnt_op += 1
        x_old = x_new

        cur_A = A.dot(p)
        alpha = r.dot(p) / p.dot(cur_A)
        x_new = x_old - alpha * p

        r = r - alpha*cur_A
        p[j] = 0
        print("alpha: ", alpha, " r: ", np.sum(r), " intr: ", cnt_op)
    return x_new, cnt_op


def conjugate_simple_grad(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0],))
    p = r_new = b
    while np.sum(r_new**2) > prec:
        cnt_op += 1
        x_old = x_new
        r_old = r_new

        alpha = r_old.dot(r_old)/(p.dot(A.dot(p)))
        x_new = x_old + alpha*p

        r_new = b - A.dot(x_new)
        beta = r_new.dot(r_new)/(r_old.dot(r_old))
        p = r_new + beta*p
        print("alpha: ", alpha, " r: ", np.sum(r_new), " intr: ", cnt_op)
    return x_new, cnt_op


def conjugate_smart_grad(A, b, prec):
    cnt_op = 0
    x_new = np.zeros((A.shape[0],))
    p = r_new = b
    while np.sum(r_new**2) > prec:
        cnt_op += 1
        x_old = x_new
        r_old = r_new

        A_cur = A.dot(p)
        alpha = r_old.dot(r_old) / p.dot(A_cur)
        x_new = x_old + alpha * p

        r_new = r_old - alpha * A_cur
        beta = r_new.dot(r_new) / (r_old.dot(r_old))
        p = r_new + beta * p
        print("alpha: ", alpha, " r: ", np.sum(r_new), " intr: ", cnt_op)
    return x_new, cnt_op


def gen_simm_pol_matrix(n, l=0, r=1):
    m = np.random.rand(n, n) * (r - l) + l
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i]
    A = m.dot(np.transpose(m))
    print("A\n", A)
    assert (np.all(np.linalg.eigvals(A) > 0))
    return A


def create_random_vector(n, l=0, r=1):
    return np.random.random(n) * (r - l) + l


if __name__ == "__main__":
    n = 10
    A = gen_simm_pol_matrix(n)
    x = create_random_vector(n)
    b = np.dot(A, x)

    prec = 0.00001
    print("simple_grad:")
    ans1, cnt_op1 = find_grad_simple(A, b, prec)
    print("free_grad:")
    ans2, cnt_op2 = free_grad(A, b, prec)
    print("conjugate_smart_grad:")
    ans3, cnt_op3 = conjugate_smart_grad(A, b, prec)
    print("conhugate_grad:")
    ans4, cnt_op4 = conjugate_simple_grad(A, b, prec)
    print("smart_grad:")
    ans5, cnt_op5 = find_grad_smart(A, b, prec)
    print("free_grad_smart:")
    ans6, cnt_op6 = free_grad_smart(A, b, prec)

    print("simple_grad:", np.sum((ans1 - x)**2), cnt_op1)
    print("smart_grad:", np.sum((ans5 - x)**2), cnt_op5)
    print("free_grad:", np.sum((ans2 - x)**2), cnt_op2)
    print("free_grad_smart:", np.sum((ans6 - x)**2), cnt_op6)
    print("conjugate_grad:", np.sum((ans4 - x)**2), cnt_op4)
    print("conjugate_smart_grad:", np.sum((ans3 - x)**2), cnt_op3)

