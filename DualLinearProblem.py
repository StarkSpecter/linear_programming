import numpy as np

from InverseMatrix import invert_modified_matrix
from SimplexMainPhase import simplex_main_phase
from SimplexInitialPhase import initial_phase


def dual_simplex(A, b, c, B):
    m, n = A.shape
    B = list(B)
    # начальный базис A_inv
    A_B = A[:, B]
    A_inv_B = np.linalg.inv(A_B)

    while True:
        c_B = c[B]
        y = c_B.dot(A_inv_B)         # базисный допустимый план двойственной задачи
        kappa_B = A_inv_B.dot(b)     # псевдоплан (базисные)
        # полный псевдоплан
        kappa = np.zeros(n)
        for idx, bi in enumerate(B):
            kappa[bi] = kappa_B[idx]

        # если псевдоплан неотрицателен, найден оптимальный план прямой задачи
        if np.all(kappa >= -1e-12):
            return kappa, B

        # выбираем первую отрицательную компоненту псевдоплана
        k = int(np.where(kappa_B < 0)[0][0])
        delta_y = A_inv_B[k, :]      # k-я строка A_inv_B

        # вычисляем mu_j для небазисных
        mu = {}
        for j in range(n):
            if j not in B:
                mu[j] = delta_y.dot(A[:, j])
        # если все mu_j >= 0, задача не совместна
        if all(mu_j >= -1e-12 for mu_j in mu.values()):
            raise ValueError("Прямая задача не совместна")

        # вычисляем sigma_j для mu_j < 0
        sigma = {}
        for j, mu_j in mu.items():
            if mu_j < 0:
                sigma[j] = (c[j] - A[:, j].dot(y)) / mu_j
        # выбираем j0 с минимальным sigma
        j0 = min(sigma, key=lambda j: sigma[j])

        # обновляем базис
        B[k] = j0
        # обновляем обратную матрицу методо из лабы 1
        A_inv_B = invert_modified_matrix(A_inv_B, A[:, j0], k+1)

# Примеры использования

if __name__ == "__main__":
    # Пример lab3
    print("=== Начальная фаза (lab3) ===")
    c3 = np.array([1, 0, 0], dtype=float)
    A3 = np.array([[1, 1, 1], [2, 2, 2]], dtype=float)
    b3 = np.array([0, 0], dtype=float)
    x3, B3, A3_fin, b3_fin = initial_phase(A3, b3, c3)
    print("Допустимый план x:", x3)
    print("Базис B:", B3)

    # Пример lab4
    print("\n=== Двойственный симплекс (lab4) ===")
    c4 = np.array([-4, -3, -7, 0, 0], dtype=float)
    A4 = np.array([[-2, -1, -4, 1, 0],
                   [-2, -2, -2, 0, 1]], dtype=float)
    b4 = np.array([-1, -1.5], dtype=float)
    B4 = [3, 4]  # 0-based индексы базисных переменных (столбцы 4 и 5)
    x_opt, B_opt = dual_simplex(A4, b4, c4, B4)
    print("Оптимальный план x*:", x_opt)
    print("Окончательный базис B:", B_opt)
