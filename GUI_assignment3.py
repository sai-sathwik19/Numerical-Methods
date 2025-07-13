
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sympy import Function, dsolve, Eq, Derivative, symbols, solve, latex 
import sympy as sp

# Show app title and description.
st.set_page_config(page_title="Assignment 3")
st.title("Ordinary differential equations")
st.write(
    '''  '''
)

st.subheader("Choose N and P")
with st.form("form"):
    P_in = st.number_input('Enter a value of P', -100.000, 100.000, -4.000, 0.001)
    points = st.number_input("Choose number of datapoints (N)", 100, 1000, 500, 10)
    submitted = st.form_submit_button("GO")


if submitted :

    st.markdown("""
# Explicit Euler Method

## 1. The ODE Problem
We aim to numerically approximate the solution to a first-order Ordinary Differential Equation (ODE) of the form:

$$
\\frac{dy}{dt} = f(t, y),
$$

where:
- \( y \) is the unknown function of \( t \) that we want to find.
- \( f(t, y) \) defines the rate of change of \( y \) with respect to \( t \).

## 2. Explicit Euler Method
The Explicit Euler method provides a straightforward approach to this problem. It approximates the solution at the next time step \( (n+1) \) using the following formula:

$$
y_{n+1} = y_n + h f(t_n, y_n),
$$

where:
- \( h \) is the time step (a small positive value).
- \( t_n \) represents the current time point.
- \( y_n \) is the approximated solution at time \( t_n \).
- \( f(t_n, y_n) \) is the value of the derivative function at the current time and solution.

## 3. Iterative Process
The method is iterative. Starting from an initial condition \( (y_0 \) at \( t_0) \), we repeatedly apply the formula above to calculate the solution at subsequent time points:

1. **Initialization:** Begin with \( y_0 \) and \( t_0 \).
2. **Iteration:** Calculate \( y_{n+1} \) using the formula \( y_{n+1} = y_n + h f(t_n, y_n) \).
3. **Update:** Set \( t_{n+1} = t_n + h \) to move to the next time point.
4. **Repeat:** Steps 2 and 3 are repeated until the desired final time is reached.

## 4. Stability Considerations
While simple to implement, the Explicit Euler method has limitations:

- **Stability:** It can become unstable for stiff ODEs (ODEs with rapidly changing solutions) unless the time step \( (h) \) is kept very small. Instability leads to inaccurate results.
- **Accuracy:** The method's accuracy is limited, particularly for larger time steps. It is a first-order method, meaning its error is proportional to \( h \).
""", unsafe_allow_html=True)


    st.subheader ("Resolving into IVPs and solving using shooting method, Explicit Euler")
    Y = np.linspace(0, 1, points)
    h = 1 / points
    P_values = [-2, 0, 2, 5, 10]

    def explicit_euler(s, P):
        u = np.zeros(points)
        u1 = np.zeros(points)
        u[0] = 0 
        u1[0] = s  
        
        for i in range(points-1): 
            u[i+1] = u[i] + h * u1[i]
            u1[i+1] = u1[i] - h * P
        
        return u, u1

    def s_explicit(s, P):
        u, _ = explicit_euler(s, P)
        return u[-1] - 1

    fig, ax = plt.subplots (2,1,figsize = (12,15))
    fig.patch.set_facecolor("white")

    def plot_P_expl (P_values, i):
        for P in P_values:
            slope = fsolve(s_explicit, 0, args=(P))[0]
            print("slope: ", slope)
            u_final, _ = explicit_euler(slope, P)
            ax[i].plot(u_final, Y, label=f'P = {P}')
    
    plot_P_expl ([P_in], 0); plot_P_expl (P_values, 1)

    for i in range (2):
        ax[i].tick_params(axis="x", colors="black")
        ax[i].tick_params(axis="y", colors="black")
        ax[i].spines['bottom'].set_color('black')
        ax[i].spines['left'].set_color('black') 
        ax[i].set_facecolor("white")
        ax[i].set_xlabel ('u(y)', color="black")
        ax[i].set_ylabel ('y', color="black")
        ax[i].set_title ('IVP using Explicit Euler', color="black")
        ax[i].legend ()

    st.pyplot (fig)

    st.subheader ("Reolving into IVPs and solving using shooting method, Implicit Euler")

# Displaying the Newton-Raphson Method explanation for Implicit Euler
    import streamlit as st

    st.markdown("""
# Newton-Raphson Method in Implicit Euler for Solving ODEs

## 1. Implicit Euler Method for ODEs  
Given a first-order ordinary differential equation (ODE) of the form:  
$$
\\frac{dy}{dt} = f(t, y),
$$  
the implicit Euler method approximates the solution at the next time step \( n+1 \) as:  
$$
y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}),
$$  
where \( h \) is the time step.

## 2. Nonlinear Equation  
The implicit nature of this method means that \( y_{n+1} \) appears on both sides of the equation, resulting in a nonlinear equation:  
$$
G(y_{n+1}) = y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}) = 0.
$$  
To find \( y_{n+1} \), we need to solve this nonlinear equation for \( G(y_{n+1}) = 0 \).

## 3. Newton-Raphson Method  
The Newton-Raphson method iteratively refines an initial guess \( y_{n+1}^{(0)} \) using the formula:  
$$
y_{n+1}^{(k+1)} = y_{n+1}^{(k)} - \\frac{G\\left(y_{n+1}^{(k)}\\right)}{G'\\left(y_{n+1}^{(k)}\\right)},
$$  
where \( G' \) is the derivative of \( G \) with respect to \( y_{n+1} \).
""", unsafe_allow_html=True)
    def implicit_euler(s, P):
        u = np.zeros(points)
        u1 = np.zeros(points)
        u[0] = 0
        u1[0] = s
        for i in range(points-1):
            u1_new = u1[i] - h * P
            u_new = u[i] + h * u1_new
            u[i+1] = u_new
            u1[i+1] = u1_new
        return u, u1

    def s_implicit(s, P):
        u, _ = implicit_euler(s, P)
        return u[-1] - 1  

    fig, ax = plt.subplots (2,1,figsize = (12,15))
    fig.patch.set_facecolor("white")

    def plot_P_impl (P_values, i):
        for P in P_values:
            slope = fsolve(s_implicit, 0.5, args=(P))[0]
            print(slope)
            u_final, _ = implicit_euler(slope, P)
            ax[i].plot(u_final, Y, label=f'P = {P}')

    plot_P_impl ([P_in], 0); plot_P_impl (P_values, 1)

    for i in range (2):
        ax[i].tick_params(axis="x", colors="black")
        ax[i].tick_params(axis="y", colors="black")
        ax[i].spines['bottom'].set_color('black')
        ax[i].spines['left'].set_color('black') 
        ax[i].set_facecolor("white")
        ax[i].set_xlabel ('u(y)', color="white")
        ax[i].set_ylabel ('y', color="black")
        ax[i].set_title ('IVP using Implicit Euler', color="black")
        ax[i].legend ()

    st.pyplot(fig)

    st.subheader ("Calculating Jacobi Matrix")

    v = sp.symbols('v')
    u = sp.symbols('u')
    dudt = v
    dvdt = -P_in

    def Jacobian(dudt,dvdt):
        dudt_u = sp.diff(dudt, u)
        dudt_v = sp.diff(dudt, v)
        dvdt_u = sp.diff(dvdt, u)
        dvdt_v = sp.diff(dvdt, v)
        J = np.array ([[dudt_u, dudt_v], [dvdt_u, dvdt_v]], dtype = float)

        return J
    
    J = Jacobian (dudt,dvdt)
    st.table (J)

    def lud(a):
        n = a.shape[0]
        l = np.zeros((n, n))
        u = np.zeros((n, n))
        np.fill_diagonal(l, 1)
        u[0] = a[0]

        for i in range(1, n):
            for j in range(n):
                if i <= j:
                    u[i][j] = a[i][j] - sum(u[k][j] * l[i][k] for k in range(i))
                if i > j:
                    l[i][j] = (a[i][j] - sum(u[k][j] * l[i][k] for k in range(j))) / u[j][j]
                    
        return l, u
        
    def shift(A):
        possible_shift_vals = []
        
        for i in range(np.shape(A)[0]):
            up_lim = A[i][i]
            low_lim = A[i][i] 
            
            for j in range(np.shape(A)[0]):
                if i != j :
                    up_lim=up_lim+abs(A[i][j])
                    low_lim=low_lim-abs(A[i][j])
                    
            possible_shift_vals.append(up_lim )
            possible_shift_vals.append(low_lim)    

        shift=np.max(np.abs(possible_shift_vals))
        return shift

    def UL_eigen (A, iters= 50000, tol = 1e-15):
        m,n = A.shape 
        I = np.identity (np.shape(A)[0])
        shift_A = shift(A) + 1
        A = A + I * (shift_A)
        
        D1 = A ; D2 = np.ones(np.shape(A))
        iter = 0
    
        while (np.allclose(np.diagonal (D1), np.diagonal (D2), tol)==False) :
            L,U = lud(D1)
            D2 = np.matmul (U,L)
            
            if (np.allclose(np.diagonal (D1), np.diagonal (D2), tol)==True):
                return np.diagonal(D2) -(shift_A)
                
            D1 = D2
            D2 = np.zeros((m,n))
            iter = iter + 1

            if (iter > iters):
                raise ValueError ("System fails to converge after 50000 iterations. Try another matrix")
                return "NA"

    st.write (f"The eigenvalues this matrix calculated by UL Method are: {UL_eigen(J)}")
            
    P_values = [-2,0,2,5,10]

    plt.figure(figsize=(10, 5))

   
    st.subheader ("Solving BVP")
   
    y = np.linspace(0,1,points)
    delta_y = y[1] - y[0]
    P_values = [-2,0,2,5,10]

    N = points
    A = np.zeros((N,N))
    np.fill_diagonal(A, -2)
    for i in range(N):
        for j in range(N):
            if np.abs(i-j) == 1:
                A[i][j] = 1
                A[j][i] = 1 
    A[0][0] = 1 ; A[0][1] = 0
    A[-1][-1] = 1 ; A[-1][-2] = 0

    fig, ax = plt.subplots (2,1,figsize = (12,15))
    fig.patch.set_facecolor("white")


    def plot_P (P_values, i):
        for P in P_values:
            b = np.ones(N)
            u = np.zeros(N)
            b = (-P * (delta_y)**2)*b
            b[0] = 0 ; b[-1] = 1
            A_inverse = np.linalg.inv(A)
            u = np.matmul(A_inverse, b)
            ax[i].plot(u, y, label = f'P = {P}')
    
    plot_P ([P_in], 0); plot_P (P_values, 1)

    for i in range (2):
        ax[i].tick_params(axis="x", colors="black")
        ax[i].tick_params(axis="y", colors="black")
        ax[i].spines['bottom'].set_color('black')
        ax[i].spines['left'].set_color('black') 
        ax[i].set_facecolor("white")
        ax[i].set_xlabel ('u(y)', color="black")
        ax[i].set_ylabel ('y', color="black")
        ax[i].set_title ('BVP using finite difference', color="black")
        ax[i].legend ()

    st.pyplot (fig)

    st.subheader( "SOLUTION BY ANALYTICAL METHODS")
    
    def analytical (P_input, n):
        for P in P_input:
            x = symbols('x')
            f = Function('f')
            ode = Eq(Derivative(f(x), x, x) + P, 0)

            general_solution = dsolve(ode)
            boundary_conditions = {f(0): 0, f(1): 1}
            
            constants = solve([general_solution.rhs.subs(x, 0) - boundary_conditions[f(0)],
                            general_solution.rhs.subs(x, 1) - boundary_conditions[f(1)]])
                            
            particular_solution = general_solution.subs(constants)
            st.write(f"P = {P}, The solution derived analytically : ")
            st.latex(f"{sp.latex(particular_solution)}")

            x_ = np.linspace(0,1,points)
            y_ = np.zeros(len(x_))
            for i in range(len(x_)):
                y_[i] = particular_solution.rhs.subs(x,x_[i])
            
            ax[n].plot(y_, x_, label = f'P = {P}')

    fig, ax = plt.subplots (2,1,figsize = (12,15))
    fig.patch.set_facecolor("white")
    analytical ([int(P_in)], 0); analytical (P_values, 1)

    for i in range (2):
        ax[i].tick_params(axis="x", colors="black")
        ax[i].tick_params(axis="y", colors="black")
        ax[i].spines['bottom'].set_color('black')
        ax[i].spines['left'].set_color('black') 
        ax[i].set_facecolor("white")
        ax[i].set_xlabel ('u(y)', color="white")
        ax[i].set_ylabel ('y', color="black")
        ax[i].set_title ('Analytical solution', color="black")
        ax[i].legend ()

    st.pyplot (fig)