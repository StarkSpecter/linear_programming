import numpy as np


def invert_modified_matrix(A_inv, x, i):

    ell = np.dot(A_inv, x)

    if np.abs(ell[i - 1]) < 1e-12:
        return None

    tilde_ell = ell.copy()
    tilde_ell[i - 1] = -1

    hat_ell = - (1 / ell[i - 1]) * tilde_ell

    n = A_inv.shape[0]
    Q = np.eye(n)
    Q[:, i - 1] = hat_ell

    new_A_inv = np.empty((n, n))
    for j in range(n):
        for k in range(n):
            if j == i - 1:
                new_A_inv[j, k] = hat_ell[i - 1] * A_inv[i - 1, k]
            else:
                new_A_inv[j, k] = A_inv[j, k] + hat_ell[j] * A_inv[i - 1, k]

    return new_A_inv


def main():

    A = np.array([[1, -1, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)

    try:
        A_inv = np.linalg.inv(A)
    except:
        print("Исходная матрица не обратима")

    x = np.array([0, 0, 0], dtype=float)

    i = 3

    new_A_inv = invert_modified_matrix(A_inv, x, i)

    if new_A_inv is None:
        print("Матрица A необратима (ℓ[{}] == 0).".format(i))
    else:
        print("Матрица A обратима.")
        print("\nНовое обратное значение (A)⁻¹:")
        print(new_A_inv)

        A_new = A.copy()
        A_new[:, i - 1] = x
        print("\nПроверка: произведение A_new * (A)⁻¹ (ожидается единичная матрица):")
        print(np.dot(A_new, new_A_inv))


if __name__ == "__main__":
    try:
        main()
    except:
        print("В процессе выполнения произошла ошибка")
