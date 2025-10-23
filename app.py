
import base64
from pathlib import Path

import numpy as np
import streamlit as st

from methods.gaussian import (
    GaussianEliminationError,
    eliminacao_gauss_pivoteamento_parcial,
    ler_matriz,
    ler_vetor,
    matriz_aumentada_para_str,
)
from methods.root_finding import RootFindingError, construir_funcao, falsa_posicao, secante

st.set_page_config(page_title="Hub de Métodos Numéricos", page_icon="🧮", layout="centered")
BASE_DIR = Path(__file__).resolve().parent
IMAGENS_DIR = BASE_DIR / "Images"


def _imagem_base64(path: Path) -> str:
    with path.open("rb") as imagem:
        return base64.b64encode(imagem.read()).decode("utf-8")


THEMES = {
    "Eliminação de Gauss (Pivoteamento Parcial)": {
        "nome": "Dr. Facilier",
        "slogan": "Truques sombrios para dominar sistemas lineares.",
        "imagem": _imagem_base64(IMAGENS_DIR / "dr_facilier.png"),
        "background": "linear-gradient(135deg, #090314 0%, #2C1157 45%, #05020B 100%)",
        "texto": "#F8EAFF",
        "painel_bg": "rgba(28, 12, 54, 0.75)",
        "painel_borda": "rgba(241, 196, 15, 0.55)",
        "painel_texto": "#F8EAFF",
        "botao_bg": "#F1C40F",
        "botao_texto": "#1E1433",
        "entrada_bg": "rgba(20, 9, 43, 0.55)",
        "entrada_borda": "rgba(241, 196, 15, 0.45)",
    },
    "Falsa Posição": {
        "nome": "Scar",
        "slogan": "Astúcia felina para encontrar raízes com segurança.",
        "imagem": _imagem_base64(IMAGENS_DIR / "scar_vilao.png"),
        "background": "linear-gradient(135deg, #0C1404 0%, #3A1E07 40%, #060B04 100%)",
        "texto": "#FDE68A",
        "painel_bg": "rgba(32, 39, 15, 0.75)",
        "painel_borda": "rgba(250, 204, 21, 0.55)",
        "painel_texto": "#FEF9C3",
        "botao_bg": "#F97316",
        "botao_texto": "#1F1304",
        "entrada_bg": "rgba(23, 28, 12, 0.55)",
        "entrada_borda": "rgba(250, 204, 21, 0.4)",
    },
    "Secante": {
        "nome": "Úrsula",
        "slogan": "Conduza as ondas numéricas com o poder da secante.",
        "imagem": _imagem_base64(IMAGENS_DIR / "ursula_vilã.png"),
        "background": "linear-gradient(135deg, #050823 0%, #311B6B 45%, #040619 100%)",
        "texto": "#E0E9FF",
        "painel_bg": "rgba(21, 25, 68, 0.75)",
        "painel_borda": "rgba(168, 85, 247, 0.5)",
        "painel_texto": "#E0E9FF",
        "botao_bg": "#A855F7",
        "botao_texto": "#0B061A",
        "entrada_bg": "rgba(13, 16, 41, 0.55)",
        "entrada_borda": "rgba(168, 85, 247, 0.45)",
    },
}


BASE_STYLE = """
<style>
:root {
    --tema-bg: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    --tema-texto: #f8fafc;
    --tema-painel-bg: rgba(15, 23, 42, 0.65);
    --tema-painel-borda: rgba(148, 163, 184, 0.45);
    --tema-painel-texto: #f8fafc;
    --tema-botao-bg: #f97316;
    --tema-botao-texto: #111827;
    --tema-entrada-bg: rgba(15, 23, 42, 0.45);
    --tema-entrada-borda: rgba(148, 163, 184, 0.45);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--tema-bg);
    color: var(--tema-texto);
}

.stApp {
    background: transparent;
    color: var(--tema-texto);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--tema-texto);
}

.step-box {
    border: 1px solid var(--tema-painel-borda);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    background: var(--tema-painel-bg);
    color: var(--tema-painel-texto);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25);
}

.aug {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
                 "Courier New", monospace;
    white-space: pre;
    font-size: 0.95rem;
    color: var(--tema-painel-texto);
}

.stButton > button {
    background: var(--tema-botao-bg);
    color: var(--tema-botao-texto);
    font-weight: 600;
    border-radius: 999px;
    border: none;
    padding: 0.45rem 1.6rem;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.25);
}

.stButton > button:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}

.stButton > button:focus {
    outline: 2px solid var(--tema-botao-bg);
}

textarea, input, select {
    background: var(--tema-entrada-bg) !important;
    color: var(--tema-texto) !important;
    border-radius: 10px !important;
    border: 1px solid var(--tema-entrada-borda) !important;
}

label, .stSlider, .stNumberInput, .stTextInput, .stTextArea {
    color: var(--tema-texto) !important;
}

details {
    background: var(--tema-entrada-bg);
    border-radius: 12px;
    border: 1px solid var(--tema-entrada-borda);
    color: var(--tema-texto);
}

details summary {
    color: var(--tema-texto);
}

[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.35);
}

[data-testid="stSidebar"] *, [data-testid="stSidebar"] label {
    color: var(--tema-texto) !important;
}

div[data-testid="stDataFrame"] {
    background: var(--tema-painel-bg);
    border-radius: 12px;
    border: 1px solid var(--tema-painel-borda);
}

div[data-testid="stDataFrame"] * {
    color: var(--tema-painel-texto) !important;
}

.tema-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

.tema-hero img {
    width: min(320px, 100%);
    border-radius: 18px;
    border: 2px solid var(--tema-painel-borda);
    box-shadow: 0 20px 35px rgba(0, 0, 0, 0.35);
}

.tema-hero__badge {
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-size: 0.75rem;
    color: var(--tema-painel-texto);
    opacity: 0.85;
}

.tema-hero__slogan {
    font-size: 1rem;
    max-width: 420px;
    color: var(--tema-painel-texto);
}

.good {color:#16a34a}
.warn {color:#fbbf24}
</style>
"""

st.markdown(BASE_STYLE, unsafe_allow_html=True)


def aplicar_tema(config: dict) -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --tema-bg: {config['background']};
            --tema-texto: {config['texto']};
            --tema-painel-bg: {config['painel_bg']};
            --tema-painel-borda: {config['painel_borda']};
            --tema-painel-texto: {config['painel_texto']};
            --tema-botao-bg: {config['botao_bg']};
            --tema-botao-texto: {config['botao_texto']};
            --tema-entrada-bg: {config['entrada_bg']};
            --tema-entrada-borda: {config['entrada_borda']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("🧮 Hub de Métodos Numéricos")
st.caption("Escolha um método para resolver seu problema.")

# ---- Seções de páginas ----


def render_eliminacao_gauss(tema: dict) -> None:
    st.subheader("Eliminação de Gauss com Pivoteamento Parcial")
    st.caption(
        "Digite a matriz A (NxN) e o vetor b. O app aplica pivoteamento parcial, mostra cada passo e a retrossubstituição."
    )

    st.markdown(
        f"""
        <div class="tema-hero">
            <span class="tema-hero__badge">Tema: {tema['nome']}</span>
            <img src="data:image/png;base64,{tema['imagem']}" alt="{tema['nome']}">
            <div class="tema-hero__slogan">{tema['slogan']}</div>
        </div>
        """,
        unsafe_allow_html=True,
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


def render_falsa_posicao(tema: dict) -> None:
    st.subheader("Método da Falsa Posição (Regula Falsi)")
    st.caption("Informe f(x) e um intervalo [a, b] inicial tal que f(a)·f(b) < 0.")

    st.markdown(
        f"""
        <div class="tema-hero">
            <span class="tema-hero__badge">Tema: {tema['nome']}</span>
            <img src="data:image/png;base64,{tema['imagem']}" alt="{tema['nome']}">
            <div class="tema-hero__slogan">{tema['slogan']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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


def render_secante(tema: dict) -> None:
    st.subheader("Método da Secante")
    st.caption(
        "Informe f(x) e duas aproximações iniciais. O método não exige mudança de sinal no intervalo."
    )

    st.markdown(
        f"""
        <div class="tema-hero">
            <span class="tema-hero__badge">Tema: {tema['nome']}</span>
            <img src="data:image/png;base64,{tema['imagem']}" alt="{tema['nome']}">
            <div class="tema-hero__slogan">{tema['slogan']}</div>
        </div>
        """,
        unsafe_allow_html=True,
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

tema_atual = THEMES[selected_method]
aplicar_tema(tema_atual)
_RENDERERS[selected_method](tema_atual)
