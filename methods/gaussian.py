import numpy as np


class GaussianEliminationError(Exception):
    """Erro disparado quando a eliminação de Gauss não pode prosseguir."""


def ler_matriz(texto: str) -> np.ndarray:
    """Converte linhas de números em uma matriz quadrada do NumPy."""
    rows = []
    for line in texto.strip().splitlines():
        if not line.strip():
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        rows.append([float(p) for p in parts])
    if not rows:
        return np.zeros((0, 0))
    n_cols = {len(r) for r in rows}
    if len(n_cols) != 1:
        raise GaussianEliminationError("Todas as linhas devem ter o mesmo número de colunas.")
    n = n_cols.pop()
    if len(rows) != n:
        raise GaussianEliminationError("A matriz deve ser quadrada (NxN).")
    return np.array(rows, dtype=float)


def ler_vetor(texto: str) -> np.ndarray:
    """Converte números em texto em um vetor do NumPy."""
    parts = [p for p in texto.replace(",", " ").split() if p]
    if not parts:
        return np.zeros((0,))
    return np.array([float(p) for p in parts], dtype=float)


def matriz_aumentada_para_str(A: np.ndarray, b: np.ndarray, precisao: int = 6) -> str:
    """Gera a representação textual da matriz aumentada [A|b]."""
    A_str = np.array2string(A, precision=precisao, suppress_small=True)
    b_str = np.array2string(b.reshape(-1, 1), precision=precisao, suppress_small=True)

    A_lines = A_str.splitlines()
    b_lines = b_str.splitlines()

    if len(A_lines) == 1 and A.shape[0] > 1:
        A_lines = [
            np.array2string(A[i, :], precision=precisao, suppress_small=True)
            for i in range(A.shape[0])
        ]
        b_lines = [
            np.array2string(b[i : i + 1].reshape(1, 1), precision=precisao, suppress_small=True)
            for i in range(b.shape[0])
        ]

    out = []
    for i in range(len(A_lines)):
        left = A_lines[i]
        right = b_lines[i] if i < len(b_lines) else ""
        out.append(f"{left} | {right}")
    return "\n".join(out)


def eliminacao_gauss_pivoteamento_parcial(
    A_in: np.ndarray,
    b_in: np.ndarray,
    tol: float = 1e-12,
):
    """Executa eliminação de Gauss com pivoteamento parcial."""
    if A_in.ndim != 2 or A_in.shape[0] != A_in.shape[1]:
        raise GaussianEliminationError("A deve ser uma matriz quadrada (NxN).")
    if b_in.ndim != 1 or b_in.shape[0] != A_in.shape[0]:
        raise GaussianEliminationError("b deve ter N elementos (mesmo N de linhas de A).")

    A = A_in.astype(float).copy()
    b = b_in.astype(float).copy()
    n = A.shape[0]
    passos = []
    swaps = 0

    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        pivot_val = A[pivot_row, k]

        if abs(pivot_val) < tol:
            passos.append(
                {
                    "titulo": f"Coluna {k + 1}: pivô numérico nulo",
                    "descricao": (
                        "<span class='warn'>Pivô ≈ 0</span>. O sistema pode ser singular ou mal condicionado. "
                        "A execução foi interrompida."
                    ),
                    "A": A.copy(),
                    "b": b.copy(),
                }
            )
            return passos, None, None, None, swaps, False

        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
            swaps += 1
            passos.append(
                {
                    "titulo": f"Troca de linhas L{k + 1} ↔ L{pivot_row + 1} (pivoteamento parcial)",
                    "descricao": f"Pivô escolhido: |a[{pivot_row + 1},{k + 1}]| = {abs(pivot_val):.6g}.",
                    "A": A.copy(),
                    "b": b.copy(),
                }
            )
        else:
            passos.append(
                {
                    "titulo": f"Coluna {k + 1}: pivô a[{k + 1},{k + 1}] = {pivot_val:.6g} (já é o maior)",
                    "descricao": "Nenhuma troca necessária.",
                    "A": A.copy(),
                    "b": b.copy(),
                }
            )

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            if abs(m) > tol:
                A[i, k:] = A[i, k:] - m * A[k, k:]
                b[i] = b[i] - m * b[k]
                passos.append(
                    {
                        "titulo": f"L{i + 1} ← L{i + 1} − ({m:.6g})·L{k + 1}",
                        "descricao": "Zerando entradas abaixo do pivô.",
                        "A": A.copy(),
                        "b": b.copy(),
                    }
                )
            else:
                passos.append(
                    {
                        "titulo": f"Entrada a[{i + 1},{k + 1}] já é ≈ 0 (m = {m:.2e})",
                        "descricao": "Nenhuma alteração necessária nesta linha.",
                        "A": A.copy(),
                        "b": b.copy(),
                    }
                )

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < tol:
            passos.append(
                {
                    "titulo": f"Retrossubstituição: pivô na linha {i + 1} é ≈ 0",
                    "descricao": "<span class='warn'>Sistema singular ou indefinido.</span>",
                    "A": A.copy(),
                    "b": b.copy(),
                }
            )
            return passos, A, b, None, swaps, False
        s = b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])
        x[i] = s / A[i, i]
        passos.append(
            {
                "titulo": f"Retrossubstituição na linha {i + 1}",
                "descricao": (
                    f"x[{i + 1}] = (b[{i + 1}] − Σ a[{i + 1},j]·x[j]) / a[{i + 1},{i + 1}] = {x[i]:.6g}"
                ),
                "A": A.copy(),
                "b": b.copy(),
            }
        )
    return passos, A, b, x, swaps, True
