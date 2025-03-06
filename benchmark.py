import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import time

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


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

for iterations in range(160, 201, 40):
    # for iterations in range(20, 221, 20):
    x = []
    y1 = []
    y2 = []

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
