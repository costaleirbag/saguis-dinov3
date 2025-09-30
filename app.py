
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="MDE & Power Visualizer", layout="wide")

st.title("🔎 MDE & Power Visualizer (teste Z para média)")

with st.sidebar:
    st.header("Parâmetros do teste")
    test_side = st.selectbox("Tipo de teste", ["Bilateral (≠)", "Unilateral (>)", "Unilateral (<)"])
    alpha = st.slider("α (nível de significância)", 0.001, 0.20, 0.05, step=0.001)
    target_power = st.slider("Poder desejado (1-β)", 0.50, 0.99, 0.80, step=0.01)
    sigma = st.number_input("Desvio-padrão populacional (σ)", min_value=1e-9, value=1.0, step=0.1, format="%.6f")

    mode = st.radio("Resolver para:", ["MDE (dados n)", "n (dados MDE)"])

    if mode == "MDE (dados n)":
        n = st.number_input("Tamanho da amostra (n)", min_value=2, value=100, step=1)
        mde_input = None
    else:
        mde_input = st.number_input("MDE (diferença mínima detectável)", value=0.28, step=0.01, format="%.6f")
        n = None

def critical_z(alpha, side):
    if side == "Bilateral (≠)":
        return norm.ppf(1 - alpha/2)
    else:
        return norm.ppf(1 - alpha)

def power_from_delta(k, delta, side):
    # Z ~ N(delta, 1) under H1
    if side == "Bilateral (≠)":
        return (1 - norm.cdf(k - delta)) + norm.cdf(-k - delta)
    elif side == "Unilateral (>)":
        return 1 - norm.cdf(k - delta)
    else:  # Unilateral (<)
        return norm.cdf(-k - delta)

def solve_mde(alpha, target_power, sigma, n, side):
    k = critical_z(alpha, side)
    # Bisseção na não-centralidade (delta) para casar o poder
    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        pow_mid = power_from_delta(k, mid, side)
        if pow_mid >= target_power:
            hi = mid
        else:
            lo = mid
    delta = (lo + hi) / 2
    se = sigma / np.sqrt(n)
    mde = delta * se
    return mde, delta, k

def solve_n(alpha, target_power, sigma, mde, side):
    # busca binária em n
    lo, hi = 2, 10000000
    k = critical_z(alpha, side)
    while lo < hi:
        mid = (lo + hi) // 2
        se = sigma / np.sqrt(mid)
        delta = mde / se
        pow_mid = power_from_delta(k, delta, side)
        if pow_mid >= target_power:
            hi = mid
        else:
            lo = mid + 1
    n = int(lo)
    se = sigma / np.sqrt(n)
    delta = mde / se
    return n, delta, k

if mode == "MDE (dados n)":
    mde, delta, k = solve_mde(alpha, target_power, sigma, n, test_side)
else:
    n, delta, k = solve_n(alpha, target_power, sigma, mde_input, test_side)
    mde = mde_input

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Distribuições do estatístico Z sob H₀ e H₁")
    x = np.linspace(-5, 5, 1200)
    pdf_H0 = norm.pdf(x, 0, 1)
    pdf_H1 = norm.pdf(x, delta, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, pdf_H0, label="Sob H₀ ~ N(0,1)")
    ax.plot(x, pdf_H1, linestyle="--", label=f"Sob H₁ ~ N(δ,1), δ={delta:.2f}")
    ax.axvline(k, linestyle="--", label=f"Corte crítico (k={k:.2f})")

    # Sombras de α e β (aproximadas conforme lado do teste)
    if test_side == "Bilateral (≠)":
        xa1 = np.linspace(k, 5, 400)
        xa2 = np.linspace(-5, -k, 400)
        ax.fill_between(xa1, 0, norm.pdf(xa1, 0, 1), alpha=0.25, label="α/2 (cauda direita)")
        ax.fill_between(xa2, 0, norm.pdf(xa2, 0, 1), alpha=0.25, label="α/2 (cauda esquerda)")
        xb = np.linspace(-k, k, 800)
        ax.fill_between(xb, 0, norm.pdf(xb, delta, 1), alpha=0.25, label="β (não rejeita sob H₁)")
        ax.axvline(-k, linestyle="--")
    elif test_side == "Unilateral (>)":
        xa = np.linspace(k, 5, 400)
        ax.fill_between(xa, 0, norm.pdf(xa, 0, 1), alpha=0.25, label="α")
        xb = np.linspace(-5, k, 800)
        ax.fill_between(xb, 0, norm.pdf(xb, delta, 1), alpha=0.25, label="β")
    else:  # Unilateral (<)
        xa = np.linspace(-5, -k, 400)
        ax.fill_between(xa, 0, norm.pdf(xa, 0, 1), alpha=0.25, label="α")
        xb = np.linspace(-k, 5, 800)
        ax.fill_between(xb, 0, norm.pdf(xb, delta, 1), alpha=0.25, label="β")

    ax.set_xlabel("Estatístico Z")
    ax.set_ylabel("Densidade")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("Resultados")
    st.markdown(
        f'''
        **Tipo de teste:** {test_side}  
        **α (significância):** {alpha:.3f}  
        **Poder alvo (1-β):** {target_power:.3f}  
        **σ (desvio-padrão):** {sigma:.6f}
        '''
    )
    if mode == "MDE (dados n)":
        st.markdown(
            f'''
            **n (tamanho da amostra):** {n:,}  
            **MDE (diferença mínima detectável):** **{mde:.6f}**  
            **Erro-padrão (σ/√n):** {sigma/np.sqrt(n):.6f}  
            **Parâmetro de não-centralidade (δ):** {delta:.3f}  
            **k (valor crítico):** {k:.3f}
            '''
        )
    else:
        st.markdown(
            f'''
            **MDE (entrada):** {mde:.6f}  
            **n mínimo para atingir o poder:** **{n:,}**  
            **Erro-padrão (σ/√n):** {sigma/np.sqrt(n):.6f}  
            **Parâmetro de não-centralidade (δ):** {delta:.3f}  
            **k (valor crítico):** {k:.3f}
            '''
        )

st.markdown('---')
with st.expander("Notas técnicas"):
    st.markdown(
        r'''
        - Fórmulas assumem teste **Z** (σ conhecida ou amostra grande).  
        - Poder é calculado com \(Z\sim\mathcal N(\delta,1)\) sob \(H_1\).  
        - Para duas caudas, o corte é \(k=z_{1-lpha/2}\). Para uma cauda, \(k=z_{1-lpha}\).  
        - O deslocamento mínimo necessário é resolvido numericamente (bisseção) para generalidade.
        '''
    )
