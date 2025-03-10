import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import time

# number of random matrices to run calculations on
# the time for that size will be the mean of the 20 runs
samples = 20

def basic_qr(A: np.ndarray, iterations: int) -> float:
    for i in range(iterations):
        Q, R = np.linalg.qr(A)
        A = R @ Q
    return A, np.diag(A)


def hessenberg_qr(A: np.ndarray, iterations: int) -> float:
    A = sp.linalg.hessenberg(A)
    for i in range(iterations):
        Q, R = np.linalg.qr(A)
        A = R @ Q
    return A, np.diag(A)

# export pgf for insertion to paper
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# the number of iterations to run the algorithm are
# 40, 80, 120, 160 and 200 iterations
for iterations in range(40, 201, 40):
    x = []
    y1 = []
    y2 = []

    # the size of the matrix: 10, 30, 50, ... largest is 590x590
    for n in range(10, 601, 20):
        x.append(n)
        basic_sum, hessenberg_sum = 0, 0

        for i in range(samples):
            A = np.random.randn(n, n)

            start_time = time.time()
            basic_qr(A, iterations)
            end_time = time.time()
            basic_sum += end_time - start_time

            start_time = time.time()
            hessenberg_qr(A, iterations)
            end_time = time.time()
            hessenberg_sum += end_time - start_time

        y1.append(basic_sum / samples * 1000)
        y2.append(hessenberg_sum / samples * 1000)

        print("done", n)

    plt.plot(x, y1, label="Basic")
    plt.plot(x, y2, label="Hessenberg")

    plt.xlabel("Matrix size")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.title(f"With {iterations} iterations")
    # plt.show()

    plt.savefig(f"documents/{iterations}.pgf")
    plt.close()
