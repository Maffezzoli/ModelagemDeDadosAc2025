from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np


class RootFindingError(Exception):
    """Erro disparado quando não é possível iniciar a busca pela raiz."""


_ALLOWED_NAMES = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
_ALLOWED_NAMES.update({name: getattr(np, name) for name in dir(np) if not name.startswith("_")})
_ALLOWED_NAMES.update({"np": np, "math": math})


def construir_funcao(expr: str) -> Callable[[float], float]:
    """Cria uma função f(x) a partir de uma expressão em texto."""
    if not expr or not expr.strip():
        raise RootFindingError("Informe uma expressão para f(x).")
    code = compile(expr, "<expr>", "eval")

    def _f(x: float) -> float:
        local_env = dict(_ALLOWED_NAMES)
        local_env.update({"x": x})
        return float(eval(code, {"__builtins__": {}}, local_env))

    return _f


def falsa_posicao(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Dict[str, object]:
    """Implementa o método clássico da falsa posição (Regula Falsi)."""
    fa = f(a)
    fb = f(b)
    if abs(fa) < tol:
        return {
            "sucesso": True,
            "raiz": a,
            "fx": fa,
            "iteracoes": 0,
            "passos": [],
        }
    if abs(fb) < tol:
        return {
            "sucesso": True,
            "raiz": b,
            "fx": fb,
            "iteracoes": 0,
            "passos": [],
        }
    if fa * fb > 0:
        raise RootFindingError("Intervalo não contém mudança de sinal (f(a)·f(b) > 0).")

    passos: List[Dict[str, float]] = []
    x_anterior = None

    for iteration in range(1, max_iter + 1):
        x = b - fb * (b - a) / (fb - fa)
        fx = f(x)
        error = abs(fx)
        if x_anterior is not None:
            error = min(error, abs(x - x_anterior))

        passos.append(
            {
                "iteracao": iteration,
                "a": a,
                "b": b,
                "x": x,
                "fx": fx,
                "erro": error,
            }
        )

        if abs(fx) < tol or (x_anterior is not None and abs(x - x_anterior) < tol):
            return {
                "sucesso": True,
                "raiz": x,
                "fx": fx,
                "iteracoes": iteration,
                "passos": passos,
            }

        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        x_anterior = x

    return {
        "sucesso": False,
        "raiz": x,
        "fx": fx,
        "iteracoes": max_iter,
        "passos": passos,
        "mensagem": "Número máximo de iterações atingido sem convergência ao critério de tolerância.",
    }


def secante(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Dict[str, object]:
    """Implementa o método da secante para busca de raízes."""
    f0 = f(x0)
    if abs(f0) < tol:
        return {
            "sucesso": True,
            "raiz": x0,
            "fx": f0,
            "iteracoes": 0,
            "passos": [],
        }

    f1 = f(x1)
    if abs(f1) < tol:
        return {
            "sucesso": True,
            "raiz": x1,
            "fx": f1,
            "iteracoes": 0,
            "passos": [],
        }
    passos: List[Dict[str, float]] = []

    for iteration in range(1, max_iter + 1):
        denom = (f1 - f0)
        if abs(denom) < 1e-30:
            raise RootFindingError("Divisão por zero encontrada (f(x1) - f(x0) ≈ 0).")
        x2 = x1 - f1 * (x1 - x0) / denom
        f2 = f(x2)
        error = min(abs(f2), abs(x2 - x1))

        passos.append(
            {
                "iteracao": iteration,
                "x_anterior": x0,
                "x_atual": x1,
                "x_proximo": x2,
                "fx": f2,
                "erro": error,
            }
        )

        if abs(f2) < tol or abs(x2 - x1) < tol:
            return {
                "sucesso": True,
                "raiz": x2,
                "fx": f2,
                "iteracoes": iteration,
                "passos": passos,
            }

        x0, f0 = x1, f1
        x1, f1 = x2, f2

    return {
        "sucesso": False,
        "raiz": x2,
        "fx": f2,
        "iteracoes": max_iter,
        "passos": passos,
        "mensagem": "Número máximo de iterações atingido sem convergência ao critério de tolerância.",
    }
