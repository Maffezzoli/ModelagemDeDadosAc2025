
import streamlit as st
import numpy as np

from methods.gaussian import (
    GaussianEliminationError,
    eliminacao_gauss_pivoteamento_parcial,
    ler_matriz,
    ler_vetor,
    matriz_aumentada_para_str,
)
from methods.root_finding import RootFindingError, construir_funcao, falsa_posicao, secante

st.set_page_config(page_title="Hub de Métodos Numéricos", page_icon="🧮", layout="centered")

# ---- Estilos ----
st.markdown(
    """
    <style>
    .step-box {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        background: #fafafa;
        color: #111827;
    }
    .aug {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
                     "Courier New", monospace;
        white-space: pre;
        font-size: 0.95rem;
    }
    .good {color:#166534}
    .warn {color:#92400e}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧮 Hub de Métodos Numéricos")
st.caption("Escolha um método para resolver seu problema.")

# ---- Seções de páginas ----


def render_eliminacao_gauss() -> None:
    st.subheader("Eliminação de Gauss com Pivoteamento Parcial")
    st.caption(
        "Digite a matriz A (NxN) e o vetor b. O app aplica pivoteamento parcial, mostra cada passo e a retrossubstituição."
    )

    with st.expander("Configurações", expanded=False):
        precision = st.slider(
            "Precisão de exibição (casas decimais)", 0, 10, 6, key="gauss_precision"
        )

    col1, col2 = st.columns(2)
    with col1:
        A_text = st.text_area(
            "Matriz A (NxN) — use espaço ou vírgula como separador",
            value="3 2 -1\n2 -2 4\n-1 0.5 -1",
            height=140,
            key="gauss_matrix",
        )
    with col2:
        b_text = st.text_area(
            "Vetor b (N elementos)",
            value="1 -2 0",
            height=140,
            key="gauss_vector",
        )

    if st.button("Calcular sistema", type="primary"):
        try:
            A = ler_matriz(A_text)
            b = ler_vetor(b_text)
            passos, _, _, x, swaps, ok = eliminacao_gauss_pivoteamento_parcial(A, b)

            st.subheader("Matriz Aumentada Inicial [A | b]")
            st.code(matriz_aumentada_para_str(A, b, precisao=precision), language="text")

            st.subheader("Passo a passo")
            for s in passos:
                with st.container():
                    st.markdown(f"**{s['titulo']}**")
                    st.markdown(
                        f"<div class='step-box'>{s['descricao']}</div>", unsafe_allow_html=True
                    )
                    st.markdown(
                        "<div class='aug'>" + matriz_aumentada_para_str(s["A"], s["b"], precisao=precision) + "</div>",
                        unsafe_allow_html=True,
                    )

            if ok and x is not None:
                st.success(f"Solução encontrada (com {swaps} troca(s) de linha):")
                st.write(x)
                st.caption("Vetor solução x.")
                st.subheader("Verificação (A·x ≈ b)")
                st.write("A·x =", A @ x)
                st.write("b =", b)
                st.caption(
                    "Erros numéricos de ponto flutuante são esperados (ordem de 1e-15)."
                )
            else:
                st.error(
                    "Não foi possível obter solução única (sistema singular ou mal condicionado)."
                )

        except GaussianEliminationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    with st.expander("Como usar"):
        st.markdown(
            """
            - Digite **A** com N linhas e N colunas; use espaços ou vírgulas.
            - Digite **b** com N números (uma linha).
            - Clique em **Calcular sistema**. O app fará pivoteamento parcial em cada coluna,
              mostrando trocas de linha, fatores de eliminação e a matriz aumentada a cada passo.
            - Ao final, é feita a retrossubstituição e a verificação `A·x ≈ b`.
            """
        )


def render_falsa_posicao() -> None:
    st.subheader("Método da Falsa Posição (Regula Falsi)")
    st.caption("Informe f(x) e um intervalo [a, b] inicial tal que f(a)·f(b) < 0.")

    expr = st.text_input("Função f(x)", value="x**3 - x - 2", key="fp_expr")

    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Limite inferior (a)", value=1.0, key="fp_a")
        tol = st.number_input(
            "Tolerância (ε)", value=1e-6, format="%.1e", min_value=0.0, key="fp_tol"
        )
    with col2:
        b = st.number_input("Limite superior (b)", value=2.0, key="fp_b")
        max_iter = st.number_input(
            "Máximo de iterações",
            min_value=1,
            max_value=500,
            value=50,
            step=1,
            key="fp_max_iter",
        )

    if st.button("Calcular raiz (falsa posição)", type="primary"):
        try:
            f = construir_funcao(expr)
            result = falsa_posicao(f, float(a), float(b), tol or 1e-12, int(max_iter))
            passos = result.get("passos", [])

            if passos:
                table = [
                    {
                        "Iteração": s["iteracao"],
                        "a": s["a"],
                        "b": s["b"],
                        "x": s["x"],
                        "f(x)": s["fx"],
                        "Erro": s["erro"],
                    }
                    for s in passos
                ]
                st.dataframe(table, use_container_width=True)

            if result.get("sucesso"):
                st.success(
                    f"Raiz aproximada: {result['raiz']:.6g} (|f(x)| = {abs(result['fx']):.2e})"
                )
            else:
                st.warning(result.get("mensagem", "Método não convergiu."))
                st.info(
                    f"Melhor aproximação encontrada: {result['raiz']:.6g} (|f(x)| = {abs(result['fx']):.2e})"
                )

        except RootFindingError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    with st.expander("Dicas"):
        st.markdown(
            """
            - Certifique-se de que f(a) e f(b) tenham sinais opostos.
            - Ajuste a tolerância e o número máximo de iterações conforme a precisão desejada.
            - O método mantém o intervalo sempre contendo a raiz, garantindo convergência.
            """
        )


def render_secante() -> None:
    st.subheader("Método da Secante")
    st.caption(
        "Informe f(x) e duas aproximações iniciais. O método não exige mudança de sinal no intervalo."
    )

    expr = st.text_input("Função f(x)", value="x**3 - x - 2", key="sec_expr")

    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0, key="sec_x0")
        tol = st.number_input(
            "Tolerância (ε)", value=1e-6, format="%.1e", min_value=0.0, key="sec_tol"
        )
    with col2:
        x1 = st.number_input("x₁", value=1.5, key="sec_x1")
        max_iter = st.number_input(
            "Máximo de iterações",
            min_value=1,
            max_value=500,
            value=50,
            step=1,
            key="sec_max_iter",
        )

    if st.button("Calcular raiz (secante)", type="primary"):
        try:
            f = construir_funcao(expr)
            result = secante(f, float(x0), float(x1), tol or 1e-12, int(max_iter))
            passos = result.get("passos", [])

            if passos:
                table = [
                    {
                        "Iteração": s["iteracao"],
                        "xₙ₋₁": s["x_anterior"],
                        "xₙ": s["x_atual"],
                        "xₙ₊₁": s["x_proximo"],
                        "f(xₙ₊₁)": s["fx"],
                        "Erro": s["erro"],
                    }
                    for s in passos
                ]
                st.dataframe(table, use_container_width=True)

            if result.get("sucesso"):
                st.success(
                    f"Raiz aproximada: {result['raiz']:.6g} (|f(x)| = {abs(result['fx']):.2e})"
                )
            else:
                st.warning(result.get("mensagem", "Método não convergiu."))
                st.info(
                    f"Melhor aproximação encontrada: {result['raiz']:.6g} (|f(x)| = {abs(result['fx']):.2e})"
                )

        except RootFindingError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    with st.expander("Dicas"):
        st.markdown(
            """
            - Escolha aproximações iniciais próximas da raiz esperada para acelerar a convergência.
            - Se o método oscilar, tente atualizar os valores iniciais ou reduzir a tolerância.
            - A secante pode convergir mais rápido que o método da bisseção e a falsa posição, mas não garante intervalo.
            """
        )

st.sidebar.header("Métodos disponíveis")


selected_method = st.sidebar.radio(
    "Selecione um método",
    (
        "Eliminação de Gauss (Pivoteamento Parcial)",
        "Falsa Posição",
        "Secante",
    ),
)

_RENDERERS = {
    "Eliminação de Gauss (Pivoteamento Parcial)": render_eliminacao_gauss,
    "Falsa Posição": render_falsa_posicao,
    "Secante": render_secante,
}

_RENDERERS[selected_method]()
