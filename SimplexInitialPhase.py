import numpy as np

from SimplexMainPhase import simplex_main_phase


def initial_phase(A, b, c):
    m, n = A.shape
    # Шаг 1: сделать b >= 0
    A_mod = A.copy().astype(float)
    b_mod = b.copy().astype(float)
    for i in range(m):
        if b_mod[i] < 0:
            A_mod[i, :] *= -1
            b_mod[i] *= -1

    # Шаг 2: сформировать вспомогательную задачу
    A_tilde = np.hstack([A_mod, np.eye(m)])
    c_tilde = np.hstack([np.zeros(n), -np.ones(m)])

    # Шаг 3: начальный базис искусственных переменных
    B = list(range(n, n + m))

    # Шаг 4: решить вспомогательную задачу
    x_tilde = simplex_main_phase(A_tilde, b_mod, c_tilde, B)

    # Шаг 5: проверка совместности
    if np.any(x_tilde[n:] > 1e-12):
        raise ValueError("Задача не имеет допустимых планов")

    # начальный допустимый план исходной задачи
    x = x_tilde[:n]

    # Шаги корректировки базиса
    A_curr = A_mod.copy()
    b_curr = b_mod.copy()
    A_t_curr = A_tilde.copy()
    B_curr = B.copy()

    # пока в базисе есть искусственные индексы
    while any(j >= n for j in B_curr):
        # выбрать максимальный искусственный индекс
        jk = max(j for j in B_curr if j >= n)
        k = B_curr.index(jk)
        i = jk - n  # 0-based номер строки

        # построить обратную базисную матрицу для A_t_curr
        A_B = A_t_curr[:, B_curr]
        A_inv_B = np.linalg.inv(A_B)

        # попытаться заменить искусственную переменную
        replaced = False
        for j in range(n):
            if j not in B_curr:
                ell = A_inv_B.dot(A_t_curr[:, j])
                if abs(ell[k]) > 1e-12:
                    B_curr[k] = j
                    replaced = True
                    break
        if replaced:
            continue

        # иначе удалить линейно зависимое ограничение i
        A_curr = np.delete(A_curr, i, axis=0)
        b_curr = np.delete(b_curr, i)
        A_t_curr = np.delete(A_t_curr, i, axis=0)
        B_curr.remove(jk)

    return x, B_curr, A_curr, b_curr


if __name__ == "__main__":

    c = np.array([1, 0, 0], dtype=float)
    A = np.array([[1, 1, 1], [2, 2, 2]], dtype=float)
    b = np.array([0, 0], dtype=float)

    x, B_final, A_fin, b_fin = initial_phase(A, b, c)
    print("Допустимый план x:", x)
    print("Базис B (0-based):", B_final)
    print("Оставшаяся A:", A_fin)
    print("Оставшийся b:", b_fin)
