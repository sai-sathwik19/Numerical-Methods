import streamlit as st
import numpy as np
from scipy.linalg import lu, hilbert
from numpy.polynomial import polynomial as P
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Assignment 1",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .highlight {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Mathematical functions (keeping the same implementations)
def LU_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for k in range(i, n):
            U[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                L[k][i] = (A[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]
    return L, U

def eigen_values(A):
    n = len(A)
    count = 0
    temp = A.copy()
    while not np.all(np.diagonal(temp) != 0):
        temp = A + (count + 1) * np.eye(n)
        count += 1
    eigenvalues = np.zeros(n)
    while True:
        tolerance = 1e-9
        L, U = LU_decomposition(temp)
        temp = np.matmul(U, L)
        if np.all(abs(U - np.eye(n)) < tolerance):
            eigenvalues = np.diag(L) - count
            break
        if np.all(abs(L - np.eye(n)) < tolerance):
            eigenvalues = np.diag(U) - count
            break
    return eigenvalues

def condition_number(Matrix):
    eigen_values = np.linalg.eigvals(Matrix)
    return np.amax(abs(eigen_values)) / np.amin(abs(eigen_values))

def gauss_jordan_inverse(a):
    a = np.array(a, float)
    n = len(a)
    tol = 1e-8
    inverse = np.identity(n)
    
    for k in range(n):
        if np.fabs(a[k, k]) < tol:
            for i in range(k + 1, n):
                if np.fabs(a[i, k]) > np.fabs(a[k, k]):
                    a[[k, i]] = a[[i, k]]
                    inverse[[k, i]] = inverse[[i, k]]
                    break
        
        div = a[k, k]
        a[k, :] /= div
        inverse[k, :] /= div
        
        for i in range(n):
            if i == k:
                continue
            factor = a[i, k]
            a[i, :] -= factor * a[k, :]
            inverse[i, :] -= factor * inverse[k, :]
    
    return inverse

def solve_system(A, b):
    
    return np.linalg.solve(A,b)

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    return x

# Streamlit UI
st.title(" Matrix")

# Sidebar
with st.sidebar:
    st.header("⚙ Configuration")
    matrix_option = st.radio("Select Matrix Input Option", ("Default Matrix", "Custom Matrix"))

# Matrix Input Section
if matrix_option == "Default Matrix":
    A = np.array([[4, 1, 2, 1, 0], 
                  [1, 3, 0, 1, 2], 
                  [2, 0, 5, 3, 1], 
                  [1, 1, 3, 6, 2], 
                  [0, 2, 1, 2, 7]])
    
    with st.expander("📊 View Default Matrix", expanded=True):
        st.dataframe(pd.DataFrame(A), use_container_width=True)
    
    b = np.array([1, 2, 3, 4, 5])
    with st.expander("📋 View Default Vector b", expanded=True):
        st.dataframe(pd.DataFrame(b).T, use_container_width=True)

else:
    st.subheader("🎯 Custom Matrix Input")
    n = st.number_input("Matrix Size (n × n):", min_value=2, max_value=10, value=5)
    
    # Create an editable dataframe for matrix input
    if 'matrix_df' not in st.session_state:
        st.session_state.matrix_df = pd.DataFrame(np.zeros((n, n)))
    
    if 'vector_b' not in st.session_state:
        # Create vector b with explicit row labels
        st.session_state.vector_b = pd.DataFrame(
            np.zeros(n),
            index=[f'b_{i+1}' for i in range(n)],
            columns=['Value']
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Enter Matrix A:")
        edited_matrix = st.data_editor(st.session_state.matrix_df, key='matrix_editor')
        A = edited_matrix.values
    
    with col2:
        st.write("Enter Vector b:")
        edited_vector = st.data_editor(
            st.session_state.vector_b,
            key='vector_editor',
            use_container_width=True,
            disabled=['index']  # Prevent index modification
        )
        b = edited_vector['Value'].values


# Main computation options
option = st.sidebar.selectbox(
    "Choose Computation",
    ["Home",
     "A) Eigenvalues",
     "B) Determinant",
     "C) Condition Number",
     "D) Characteristic Polynomial",
     "E) Power Method",
     "F) Matrix Inverse",
     "G) Solve System"]
)

# Results section
st.markdown("---")
st.header("📊 Results")

if option == "A) Eigenvalues":
    if st.button("Calculate Eigenvalues"):
        with st.spinner('Computing eigenvalues...'):
            eigenvalues = np.linalg.eigvals(A)
            st.markdown("""
            <div class="result-box">
                <h3>📊 Eigenvalues:</h3>
            </div>
            """, unsafe_allow_html=True)
            st.write(eigenvalues)

    # LU Decomposition Method
    st.markdown(r"""
# LU Decomposition Method

LU decomposition is a matrix factorization technique that decomposes a matrix *A* into: *A = LU* where:

* *L* is a lower triangular matrix
* *U* is an upper triangular matrix

Key Applications:

1. Solving systems of linear equations
2. Finding matrix inverse
3. Computing determinants
""")                 
    
    st.markdown(r"""
# Eigenvalue Calculation with Tolerance

The iterative process for finding eigenvalues uses a tolerance ($tol$) to determine convergence:

1. Initialize matrix $A$ and tolerance $tol$ (typically $10^{-9}$)
2. Perform LU decomposition: $A = LU$
3. Update $A = U \times L$
4. Check stopping condition:
   $\|U - I\| < tol$ or $\|L - I\| < tol$

The stopping condition ensures that either $U$ or $L$ is sufficiently close to the identity matrix $I$.
The eigenvalues are found on the diagonal of either $L$ or $U$ when convergence is reached.
""")
    
    st.markdown(r"""
# Shifting Method for Eigenvalue Calculation

When a matrix has zero diagonal elements, we use the shifting method:

1. Choose a shift value $\lambda$
2. Create shifted matrix: $A_{\lambda} = A - \lambda I$
3. Find eigenvalues of $A_{\lambda}$
4. Add $\lambda$ back to get original eigenvalues

The shift transformation:
$A_{\lambda} = A - \lambda I$

Properties:
- Preserves eigenvectors
- Shifts eigenvalues by $\lambda$
- If $v$ is an eigenvalue of $A_{\lambda}$, then $v + \lambda$ is an eigenvalue of $A$

This method is particularly useful when:
- Matrix has zero diagonal elements
- Matrix is poorly conditioned
- Need to find specific eigenvalues in a certain region
""")

elif option == "B) Determinant":
    if st.button("Calculate Determinant"):
        with st.spinner('Computing determinant...'):
            eigenvalues = np.linalg.eigvals(A)
            det = np.prod(eigenvalues)
            st.markdown(f"""
            <div class="result-box">
                <h3>🎯 Determinant: {det:.4f}</h3>
                <p>{'✅ The system has unique solutions' if det != 0 else '⚠ The system does not have unique solutions'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown(r"""
$det(A) = \prod_{i=1}^n \lambda_i$

The given code determines the determinant of the given matrix using its eigenvalues,
determinant of the given matrix A is the product of all of its eigenvalues.
""")
    

elif option == "C) Condition Number":
    if st.button("Calculate Condition Numbers"):
        with st.spinner('Computing condition numbers...'):
            hilbert_matrix = hilbert(5)
            hilbert_cond = condition_number(hilbert_matrix)
            matrix_cond = condition_number(A)
            
            st.markdown(f"""
            <div class="result-box">
                <h3>📊 Condition Numbers:</h3>
                <p>Given Matrix: {matrix_cond:.2f}</p>
                <p>Hilbert Matrix: {hilbert_cond:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown(r"""
### Condition Number Formula:
$$ \text{Condition Number} = \frac{\max(|\lambda|)}{\min(|\lambda|)} $$
                
This section of code calculates and compares the condition numbers of matrix A and a 5x5 Hilbert matrix. A Hilbert matrix is a square matrix where the elements are the unit fractions. Hilbert matrices are known to be ill-conditioned, meaning that they can be sensitive to small changes in input values, leading to large errors in the output. By comparing the condition numbers, you can assess the relative conditioning of matrix A to the Hilbert matrix.

""")


elif option == "D) Characteristic Polynomial":
    if st.button("Calculate Characteristic Polynomial"):
        with st.spinner('Computing characteristic polynomial...'):
            eigenvalues = eigen_values(A)
            polynomial = np.round(P.polyfromroots(eigenvalues))
            polynomial = np.flip(polynomial)
            
            # Create algebraic form of the polynomial
            polynomial_str = ""
            degree = len(polynomial) - 1
            
            for i, coeff in enumerate(polynomial):
                power = degree - i
                if coeff != 0:
                    if i > 0 and coeff > 0:
                        polynomial_str += " + "
                    elif i > 0 and coeff < 0:
                        polynomial_str += " - "
                    elif i == 0 and coeff < 0:
                        polynomial_str += "-"
                    
                    if power == 0:
                        polynomial_str += f"{abs(coeff)}"
                    elif power == 1:
                        polynomial_str += f"{abs(coeff)}λ"
                    else:
                        polynomial_str += f"{abs(coeff)}λ^{power}"
            
            st.markdown("""
            <div class="result-box">
                <h3>🔢 Characteristic Polynomial:</h3>
            </div>
            """, unsafe_allow_html=True)
            st.latex(f"P(λ) = {polynomial_str}")
            
            st.markdown("""
            <div class="highlight">
                <h4>📊 Polynomial Coefficients:</h4>
            </div>
            """, unsafe_allow_html=True)
            st.write(polynomial)

elif option == "E) Power Method":
    def eigen_vector_power(matrix):
        n = len(matrix)
        eigen_vector = np.ones(n)  # Initial eigenvector
        max_iterations = 1000
        tolerance = 1e-6
        
        for _ in range(max_iterations):
            new_vector = np.dot(matrix, eigen_vector)
            new_vector = new_vector / np.linalg.norm(new_vector)
            
            if np.allclose(abs(new_vector), abs(eigen_vector), rtol=tolerance):
                eigenvalue = np.dot(np.dot(matrix, new_vector), new_vector)
                return eigenvalue
            
            eigen_vector = new_vector
            
        return np.dot(np.dot(matrix, eigen_vector), eigen_vector)

    if st.button("Calculate Using Power Method"):
        with st.spinner('Computing eigenvalues using power method...'):
            # Compute max eigenvalue
            max_eigenvalue = eigen_vector_power(A)
            
            # Compute min eigenvalue using inverse power method
            try:
                A_inv = np.linalg.inv(A)
                min_eigenvalue = 1 / eigen_vector_power(A_inv)
                
                st.markdown("""
                <div class="result-box">
                    <h3>📊 Power Method Results:</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="highlight">
                        <h4>Maximum Eigenvalue:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(f"{max_eigenvalue:.6f}")
                
                with col2:
                    st.markdown("""
                    <div class="highlight">
                        <h4>Minimum Eigenvalue:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(f"{min_eigenvalue:.6f}")
                
                # Compare with actual eigenvalues
                actual_eigenvalues = eigen_values(A)
                st.markdown("""
                <div class="highlight">
                    <h4>Verification with Actual Eigenvalues:</h4>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"Actual max: {np.max(actual_eigenvalues):.6f}")
                st.write(f"Actual min: {np.min(np.abs(actual_eigenvalues)):.6f}")
                
            except np.linalg.LinAlgError:
                st.error("Error: Matrix is singular, cannot compute inverse for minimum eigenvalue.")

    st.markdown("""This code calculates the absolute maximum and minimum eigenvalues of matrix A using the power method.

The power method is an iterative algorithm that finds the dominant eigenvalue (largest eigenvalue in absolute value) and its corresponding eigenvector. To find the absolute minimum eigenvalue, the power method is applied to the inverse of A. The reciprocal of the absolute maximum eigenvalue of the inverse matrix is then the absolute minimum eigenvalue of the original matrix.            
""")
def gauss_jordan_inverse(A):
    n = len(A)
    # Augment matrix A with identity matrix
    augmented = np.concatenate((A, np.identity(n)), axis=1)
    
    try:
        # Check if matrix is singular
        if np.linalg.det(A) == 0:
            raise np.linalg.LinAlgError("Matrix is singular")
            
        # Gauss-Jordan elimination
        for i in range(n):
            pivot = augmented[i][i]
            if pivot == 0:
                raise np.linalg.LinAlgError("Matrix is singular")
                
            # Divide row by pivot
            augmented[i] = augmented[i] / pivot
            
            # Eliminate column
            for j in range(n):
                if i != j:
                    augmented[j] = augmented[j] - augmented[i] * augmented[j][i]
                    
        # Extract inverse matrix
        return augmented[:, n:]
        
    except np.linalg.LinAlgError:
        return None

if option == "F) Matrix Inverse":
    if st.button("Calculate Matrix Inverse"):
        with st.spinner('Computing matrix inverse...'):
            inverse_matrix = gauss_jordan_inverse(A)
            
            if inverse_matrix is None:
                st.error("Since it's singular, inverse can't be computed")
            else:
                st.markdown("""
                <div class="result-box">
                    <h3>📊 Matrix Inverse:</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display the inverse matrix in a dataframe
                st.dataframe(
                    pd.DataFrame(
                        inverse_matrix,
                        columns=[f'Col {i+1}' for i in range(len(inverse_matrix))],
                        index=[f'Row {i+1}' for i in range(len(inverse_matrix))]
                    )
                )
                
                # Verification
                verification = np.allclose(np.dot(A, inverse_matrix), np.eye(len(A)))
                st.markdown(f"""
                <div class="highlight">
                    <h4>Verification:</h4>
                    <p>{'✅ Inverse verified (A × A⁻¹ ≈ I)' if verification else '⚠ Verification failed'}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
        This code calculates the inverse of matrix A using the Gauss-Jordan elimination method. 
        The Gauss-Jordan method is a standard algorithm for finding the inverse of a matrix by 
        transforming the augmented matrix (A|I), where I is the identity matrix, into reduced 
        row-echelon form (I|A^-1). The resulting matrix on the right side of the augmented 
        matrix is the inverse of the original matrix.
    """)
    
elif option == "G) Solve System":
    if st.button("Solve System Ax = b"):
        with st.spinner('Solving system...'):
            try:
                solution = solve_system(A, b)
                
                st.markdown("""
                <div class="result-box">
                    <h3>🎯 Solution Vector x:</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display solution in a styled dataframe
                solution_df = pd.DataFrame(solution, columns=['Value'], index=[f'x_{i+1}' for i in range(len(solution))])
                st.dataframe(solution_df.style.background_gradient(cmap='viridis'))
                
                # Verification
                verification_result = np.allclose(np.dot(A, solution), b)
                residual = np.linalg.norm(np.dot(A, solution) - b)
                
                st.markdown(f"""
                <div class="highlight">
                    <h4>Solution Verification:</h4>
                    <p>{'✅ Solution verified (Ax ≈ b)' if verification_result else '⚠ Solution may be inaccurate'}</p>
                    <p>Residual norm: {residual:.2e}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visual comparison of Ax and b
                comparison_df = pd.DataFrame({
                    'b (Original)': b,
                    'Ax (Computed)': np.dot(A, solution)
                }, index=[f'Equation {i+1}' for i in range(len(b))])
                
                st.markdown("""
                <div class="highlight">
                    <h4>Comparison of Ax and b:</h4>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(comparison_df.style.background_gradient(cmap='coolwarm'))
                
            except np.linalg.LinAlgError:
                st.error("Error: System cannot be solved (matrix may be singular)")
            except Exception as e:
                st.error(f"Error solving system: {str(e)}")
