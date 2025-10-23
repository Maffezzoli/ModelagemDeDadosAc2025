# Hub de Métodos Numéricos

Aplicativo Streamlit que reúne diferentes técnicas numéricas em uma única interface. O usuário escolhe o método desejado, fornece os parâmetros necessários e visualiza tanto o resultado final quanto os passos intermediários.

## Estrutura do projeto

```
app.py
methods/
    __init__.py
    gaussian.py
    root_finding.py
requirements.txt
```

- `app.py`: ponto de entrada do Streamlit. Renderiza o "hub" com a barra lateral de seleção e organiza as páginas de cada método.
- `methods/gaussian.py`: funções auxiliares para entrada de dados, formatação da matriz aumentada e implementação da eliminação de Gauss com pivoteamento parcial.
- `methods/root_finding.py`: utilitários para construir funções a partir de expressões, além dos algoritmos da falsa posição e da secante.

## Instalação

1. Crie e ative um ambiente virtual (opcional, porém recomendado).
2. Instale as dependências listadas em `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Execução

Rode o Streamlit diretamente a partir da raiz do projeto:

```bash
streamlit run app.py
```

A aplicação abrirá no navegador e exibirá a barra lateral com os métodos disponíveis.

## Métodos disponíveis

### Eliminação de Gauss com pivoteamento parcial
- Entrada: matriz quadrada `A` e vetor `b`.
- Funcionalidades principais:
  - `ler_matriz` e `ler_vetor`: convertem texto em `numpy.ndarray` validados.
  - `eliminacao_gauss_pivoteamento_parcial`: executa a eliminação, registra cada operação e retorna os passos, a matriz escalonada, o vetor transformado e a solução.
  - `matriz_aumentada_para_str`: formata a matriz aumentada em texto para exibição no Streamlit.
- Saída: solução do sistema, quantidade de trocas de linha e verificação `A·x ≈ b`.

### Método da falsa posição (Regula Falsi)
- Entrada: expressão para `f(x)`, intervalo `[a, b]`, tolerância e máximo de iterações.
- Funções relevantes:
  - `construir_funcao`: monta `f(x)` a partir de uma expressão em texto, expondo apenas funções seguras de `math` e `numpy`.
  - `falsa_posicao`: implementa o algoritmo clássico, registra cada iteração e verifica convergência pelo valor de `f(x)` e pela variação de `x`.
- Saída: raiz aproximada, valor de `f(x)` na raiz e tabela com os passos executados.

### Método da secante
- Entrada: expressão para `f(x)`, aproximações iniciais `x0` e `x1`, tolerância e máximo de iterações.
- Funções relevantes:
  - Reaproveita `construir_funcao` para gerar `f(x)`.
  - `secante`: calcula sucessivas aproximações usando a secante, registrando os pares `(x_n, f(x_n))` e o erro a cada passo.
- Saída: raiz aproximada, valor de `f(x)` na raiz e tabela iterativa.

## Tratamento de erros

- Os métodos disparam exceções específicas (`GaussianEliminationError` e `RootFindingError`) quando as entradas são inválidas ou algum pré-requisito não é atendido.
- O Streamlit captura essas exceções e exibe mensagens amigáveis ao usuário.

## Personalização

- Ajuste os textos padrão, limites dos sliders e aparência do aplicativo diretamente em `app.py`.
- Novos métodos podem ser adicionados criando funções equivalentes em `methods/` e registrando um novo renderizador no dicionário `_RENDERERS`.
