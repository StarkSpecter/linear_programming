import numpy as np

from InverseMatrix import invert_modified_matrix


def simplex_main_phase(A, x, c, B):
    """
    Основная фаза симплекс-метода.
    A: матрица ограничений (m×n)
    b: вектор доп плана (m)
    c: вектор коэффициентов целевой функции (n)
    B: список индексов базисных переменных (0-based, длина m), упорядоченный
    """
    m, n = A.shape
    # начальный базис
    A_B = A[:, B]
    A_inv_B = np.linalg.inv(A_B)
    # начальный план x_B = A_inv_B @ b
    x_B = A_inv_B.dot(x)

    while True:
        # 1) потенциалы u = c_B^T A_inv_B
        c_B = c[B]
        u = c_B.dot(A_inv_B)  # (1×m)
        # 2) оценки Δ = u A − c
        Delta = u.dot(A) - c  # (1×n)

        # 3) проверка оптимальности
        if np.all(Delta >= -1e-12):
            # собираем полный вектор x
            x = np.zeros(n)
            x[B] = x_B
            return x

        # 4) выбираем первый j0 с Δ[j0] < 0
        j0 = int(np.where(Delta < 0)[0][0])

        # 5)  z = A_inv_B @ A[:, j0]
        z = A_inv_B.dot(A[:, j0])

        # 6) считаем тэта_i
        thetas = np.full(m, np.inf)
        for i in range(m):
            if z[i] > 1e-12:
                thetas[i] = x_B[i] / z[i]

        theta0 = thetas.min()
        if np.isinf(theta0):
            raise ValueError("Целевая функция не ограничена сверху")

        # 7) определяем, какой базисный индекс уходит: первый argmin θ
        k = int(np.argmin(thetas))  # 0-based внутри базиса
        j_leave = B[k]

        # 8) обновляем базис: на месте k ставим j0
        B[k] = j0

        # 9) обновляем x_B:
        #    для нового базиса x_j0 = theta0, остальные x_B[i] = x_B[i] − theta0*z[i]
        x_B = x_B - theta0 * z
        x_B[k] = theta0

        # 10) обновляем A_inv_B:
        #     базисная матрица сменилась в k-й колонке на A[:, j0]
        A_inv_B = invert_modified_matrix(A_inv_B, A[:, j0], k + 1)


def main():
    # Задача: max c^T x, Ax = b, x >= 0
    c = np.array([1, 1, 0, 0, 0], dtype=float)
    A = np.array([
        [-1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
    ], dtype=float)
    x = np.array([1, 3, 2], dtype=float)
    # Начальный базис B = (3,4,5):
    B0 = [2, 3, 4]

    x_opt = simplex_main_phase(A, x, c, B0)
    print("Оптимальный план x*:")
    print(x_opt)


if __name__ == "__main__":
    main()
