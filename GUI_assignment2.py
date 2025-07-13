import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import streamlit as st
from sympy import symbols, expand, Poly

# Page configuration
st.set_page_config(
    page_title="Gauss-Legendre Polynomial Visualizer",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
        margin: 5px 0;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .main {
        padding: 2rem;
    }
    h1 {
        text-align: center;
        color: #1f77b4;
        padding-bottom: 20px;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("Gauss-Legendre Polynomial Analysis")
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        This application visualizes Gauss-Legendre polynomials and their properties.
        Select the degree of the polynomial and explore different visualizations.
    </div>
""", unsafe_allow_html=True)

# Helper functions (keeping the original implementations)
def polynomial_to_latex(coefficients):
    # [Original implementation...]
    if not coefficients:
        return "0"
    while coefficients and coefficients[0] == 0:
        coefficients.pop(0)
    
    if not coefficients:
        return "0"
    
    degree = len(coefficients) - 1
    terms = []
    
    for i, coef in enumerate(coefficients):
        if coef == 0:
            continue
            
        current_degree = degree - i
        term = ""
        if current_degree == 0:
            term = str(abs(coef))
        else:
            if abs(coef) == 1:
                term = "" if current_degree == 1 else "1"
            else:
                term = str(abs(coef))
        
        if current_degree > 0:
            term += "x"
            if current_degree > 1:
                term += f"^{current_degree}"
        
        if i == 0:
            if coef < 0:
                term = f"-{term}"
        else:
            term = f" + {term}" if coef > 0 else f" - {term}"
            
        terms.append(term)
    
    return "$" + "".join(terms) + "$"

def jacobi_matrix(n):
    beta = 0.5 / np.sqrt(1 - (2 * np.arange(1, n, dtype=float)) ** -2)
    J = np.diag(beta, -1) + np.diag(beta, 1)
    return J

def weights_and_nodes(J):
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    roots = eigenvalues
    weights = 2 * (eigenvectors[0, :]**2)
    return roots, weights

def visualization(degree, roots, weights, title="Gaussian Quadrature"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(roots, weights, color="red", s=100, alpha=0.6)
    ax.plot(roots, weights, color="blue", alpha=0.3, linestyle='--')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{title} (n={degree})', pad=20)
    ax.set_xlabel('Roots', fontsize=12)
    ax.set_ylabel('Weights', fontsize=12)
    
    # Improve plot styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return fig

def legendre_polynomial(n):
    x = symbols('x')
    if n > 16:
        Pn = legendre(n)
        return Pn.coefficients
    if n == 0:
        return [1]
    elif n == 1:
        return [1, 0]
    
    P_n_minus_2 = 1
    P_n_minus_1 = x
    for k in range(2, n + 1):
        P_n = ((2 * k - 1) * x * P_n_minus_1 - (k - 1) * P_n_minus_2) / k
        P_n_minus_2 = P_n_minus_1
        P_n_minus_1 = P_n
    
    P_n_expanded = expand(P_n)
    coefficients = Poly(P_n_expanded, x).all_coeffs()
    return [float(c) for c in coefficients]

def compute_weights(roots):
    n = len(roots)
    weights = np.zeros(n)
    for i in range(n):
        prod = 1
        for j in range(n):
            if i != j:
                prod *= (roots[i] - roots[j])
        weights[i] = 2 / ((1 - roots[i]**2) * (prod**2))
    weights = 2 * weights / np.sum(weights)
    return weights

def companion_matrix(coeffs, degree):
    if coeffs[0] != 1:
        coeffs = [c / coeffs[0] for c in coeffs]
    C = np.zeros((degree, degree))
    C[-1, :] = -np.flip(np.array(coeffs[1:]))
    for i in range(degree - 1):
        C[i, i + 1] = 1
    return C

# Sidebar UI
st.sidebar.header("Input Parameters")
st.sidebar.markdown("---")
degree = st.sidebar.number_input("Polynomial Degree", min_value=1, value=2, step=1)

st.sidebar.markdown("### Actions")
calculate_jacobi = st.sidebar.button("Calculate Jacobi Method")
calculate_companion = st.sidebar.button("Calculate Companion Method")
plot_both = st.sidebar.button("Plot Both Methods")

st.sidebar.markdown("---")
st.sidebar.markdown("Use the above buttons to calculate roots, weights, and visualize results.")

if 'jacobi_results' not in st.session_state:
    st.session_state['jacobi_results'] = None
if 'companion_results' not in st.session_state:
    st.session_state['companion_results'] = None

# Main content area with tabs
tab1, tab2 = st.tabs(["Jacobi Matrix Method", "Companion Matrix Method"])

if 'roots' not in st.session_state:
    st.session_state['roots'] = None
if 'weights' not in st.session_state:
    st.session_state['weights'] = None
if 'calculated' not in st.session_state:
    st.session_state['calculated'] = False

with tab1:
    if st.button("Calculate using Jacobi Matrix", key="jacobi"):
        with st.spinner("Calculating..."):
            st.latex(r'''
            \mathbf{A)} \quad \text{{Jacobi Matrix}} \quad J = 
            \begin{bmatrix}
            0 & \beta_1 & 0 & \cdots & 0 & 0 \\
            \beta_1 & 0 & \beta_2 & \cdots & 0 & 0 \\
            0 & \beta_2 & 0 & \ddots & \vdots & \vdots \\
            \vdots & \vdots & \ddots & \ddots & \beta_{N-2} & 0 \\
            0 & 0 & \cdots & \beta_{N-2} & 0 & \beta_{N-1} \\
            0 & 0 & \cdots & 0 & \beta_{N-1} & 0
            \end{bmatrix}
            ''')

            st.latex(r'''
            \text{where} \quad \beta_i = \frac{1}{2\sqrt{1 - (2i)^{-2}}}
            ''')

            st.markdown('''
            *Note*: All \( \alpha \)'s are zero, as used in the code, 
            which simplifies the Jacobi matrix to a symmetric tridiagonal form.
            ''')

            J = jacobi_matrix(degree)
            st.write(f"Jacobi Matrix for n={degree}:")
            st.write(J)
            
            st.markdown("### Extracting Roots and Weights from the Jacobi Matrix")

            st.markdown("""
            The **roots** (nodes) for Gaussian quadrature are obtained as the eigenvalues of the Jacobi matrix. 

            The **weights** for the quadrature are calculated using the corresponding eigenvectors:

            """)

            st.latex(r'''
            \text{weights} = 2 \times \left( \text{first element of corresponding eigenvectors} \right)^2
            ''')

            st.latex(r'''
            \text{The eigenvalues of the Jacobi matrix represent the roots (nodes) for Gaussian quadrature.}
            
            \text{weights} = 2 \times \left( \text{first element of corresponding eigenvectors} \right)^2.
            ''')
            roots , weights =weights_and_nodes(J)
            st.session_state['jacobi_results'] = (roots, weights)
            st.session_state['roots'] = roots
            st.session_state['weights'] = weights
            st.session_state['calculated'] = True
            if st.session_state['calculated']:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Roots:")
                    st.write(np.round(roots, 6))
                with col2:
                    st.write("Weights:")
                    st.write(np.round(weights, 6))
    if st.session_state['roots'] is not None:
        if st.button("Show Plot", key="plot_jacobi") :
            st.markdown("""Visualizing Gaussian Quadrature Nodes and Weights using the Jacobi matrix.""")
            fig = visualization(degree, st.session_state['roots'], st.session_state['weights'], "Jacobi Matrix Method")
            st.pyplot(fig)

if 'roots_found' not in st.session_state:
    st.session_state['roots_found'] = None
if 'weights_found' not in st.session_state:
    st.session_state['weights_found'] = None
if 'calculated_' not in st.session_state:
    st.session_state['calculated_found'] = False

with tab2:
    if st.button("Calculate using Companion Matrix", key="companion"):
        with st.spinner("Calculating..."):
            st.latex(r'''
            \mathbf{B)} \quad \text{Calculating Legendre Polynomials using Recurrence Relation}
            ''')

            st.latex(r'''
            (2n+1)xP_{n}(x) = (n+1)P_{n+1}(x) + nP_{n-1}(x)
            ''')

            st.markdown("""
            This recurrence relation is fundamental in calculating Legendre polynomials \(P_n(x)\) for any \(n\). 
            """)
            coefficients = legendre_polynomial(degree)
            C = companion_matrix(coefficients, degree)
            roots_found = np.sort(np.linalg.eigvals(C))
            weights_found = compute_weights(roots_found)

            st.session_state['companion_results'] = (roots_found, weights_found)

            
            st.write(f"The coefficients are {coefficients}")
            st.subheader("Companion Matrix")
            st.write("Matrix Form:")
            st.write(C)  # Display as a table
            
            st.markdown("### Computing Polynomial Roots using Companion Matrix Eigenvalues")

            st.latex(r'''
            \text{Given a polynomial: } P(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_1 x + a_0,
            ''')

            st.latex(r'''
            \text{The companion matrix } C \text{ is defined as:}
            ''')

            st.latex(r'''
            C = 
            \begin{bmatrix}
            0 & 1 & 0 & \cdots & 0 \\
            0 & 0 & 1 & \cdots & 0 \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            0 & 0 & 0 & \cdots & 1 \\
            -\frac{a_0}{a_n} & -\frac{a_1}{a_n} & -\frac{a_2}{a_n} & \cdots & -\frac{a_{n-1}}{a_n}
            \end{bmatrix}
            ''')

            st.latex(r'''
            \text{The roots of } P(x) \text{ are the eigenvalues of the companion matrix } C.
            ''')

            st.markdown("""
            This method provides an efficient numerical approach for finding the roots of a polynomial.
            """)
            st.session_state['roots_found'] = roots_found
            st.session_state['weights_found'] = weights_found
            st.session_state['calculated_'] = True        
            col1, col2 = st.columns(2)
            if st.session_state['calculated_']:
                with col1:
                    st.write("Roots:")
                    st.write(np.round(roots_found, 6))
                with col2:
                    st.write("Weights:")
                    st.write(np.round(weights_found, 6))
    if st.session_state['roots_found'] is not None:
        if st.button("Show Plot",key="temp") :        
            fig = visualization(degree,  st.session_state['roots_found'], st.session_state['weights_found'] , "Companion Matrix Method")
            st.pyplot(fig)
if plot_both and st.session_state['jacobi_results'] and st.session_state['companion_results']:
    roots_jacobi, weights_jacobi = st.session_state['jacobi_results']
    roots_companion, weights_companion = st.session_state['companion_results']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot Jacobi Method
    ax.scatter(roots_jacobi, weights_jacobi, color="blue", s=100, marker='o', label="Jacobi Method")
    ax.plot(roots_jacobi, weights_jacobi, color="blue", linestyle='--', alpha=0.7)

    # Plot Companion Method with slight x-offset for visibility
    ax.scatter(roots_companion + 0.02, weights_companion, color="green", s=100, marker='^', label="Companion Method")
    ax.plot(roots_companion + 0.02, weights_companion, color="green", linestyle=':', alpha=0.7)
    
    # Add titles, grid, and legend
    ax.set_title("Comparison of Gaussian Quadrature Methods", fontsize=18, pad=20)
    ax.set_xlabel("Roots", fontsize=14)
    ax.set_ylabel("Weights (Primary Y-Axis)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="upper right", fontsize=12)
    
    # Highlight individual points
    for (x, y) in zip(roots_jacobi, weights_jacobi):
        ax.annotate(f"({x:.2f}, {y:.2f})", (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=10, color="blue")

    for (x, y) in zip(roots_companion, weights_companion):
        ax.annotate(f"({x:.2f}, {y:.2f})", (x + 0.02, y), textcoords="offset points", xytext=(5, -10), ha='center', fontsize=10, color="green")
    
    st.pyplot(fig)


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        Created for Numerical Analysis visualization
    </div>
""", unsafe_allow_html=True)