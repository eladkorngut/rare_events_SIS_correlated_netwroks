import numpy as np
import os
from scipy.optimize import fsolve
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import wald
from scipy.stats import beta as beta_sci
from scipy.linalg import eigvals
import argparse


def numerical_xstar_change_eps_alpha(kavg,epsilon,alpha,net_type,N,lam):
    x = np.zeros((len(epsilon), len(alpha)))
    for i in range(len(epsilon)):
        for j in range(len(alpha)):
            xstar,p_k = numerical_xstar(kavg, epsilon[i], alpha[j], net_type, N, lam)
            x[i, j] = np.sum(xstar*p_k)
    return x


def numerical_xstar(kavg,epsilon,correlation,net_type,N,lam):
    def Phi(x,k_avg):
        return np.dot(unique_degrees * P_k, x) / k_avg

    # def equations(x, beta, k, correlation):
    #     if isinstance(x, np.ndarray):  # Check if x is an array
    #         x = x[0]  # Extract the scalar value
    #     phi_x = Phi(x)  # Recalculate phi_x for the current x
    #     return beta * k * (1 - x) * (correlation * x + (1 - correlation) * phi_x) - x

    def equations(x,beta,correlation,unique_degrees,k_avg):
        eqs = []
        for j, k in enumerate(unique_degrees):
            phi_x = Phi(x,k_avg)
            eq = beta * k * (1 - x[j]) * (correlation * x[j] + (1 - correlation) * phi_x) - x[j]
            eqs.append(eq)
        return np.array(eqs)

    # def equations(x):
    #     eqs = []
    #     for j, k in enumerate(unique_degrees):
    #         phi_x = Phi(x)
    #         eq = beta * k * (1 - x[j]) * (correlation * x[j] + (1 - correlation) * phi_x) - x[j]
    #         eqs.append(eq)
    #     return np.array(eqs)

    if net_type == 'gam':
        shape = 1 / epsilon ** 2  # Shape parameter α
        theta = kavg * epsilon ** 2  # Scale parameter θ
        k_values = np.linspace(1, N, N)
        pdf_values = gamma.pdf(k_values, a=shape, scale=theta)

        # Apply the condition N * pdf_values > 1
        condition = N * pdf_values > 1

        # Filter k_values and pdf_values based on the condition
        unique_degrees = k_values[condition]
        P_k = pdf_values[condition]
        k_avg = np.sum(P_k*unique_degrees)
    elif net_type=='ig':
        # wald_mu, wald_lambda = kavg, kavg / epsilon ** 2
        k_values = np.linspace(1, N, N)
        pdf_values = wald.pdf(k_values, loc=0, scale=epsilon*kavg)  # Using loc and scale for the PDF
        k_avg = np.sum(pdf_values * k_values)
        pdf_values = wald.pdf(k_values, loc=kavg - k_avg, scale=epsilon * kavg)  # Using loc and scale
        # Apply the condition N * pdf_values > 1
        condition = N * pdf_values > 1

        # Filter k_values and pdf_values based on the condition
        unique_degrees = k_values[condition]
        P_k = pdf_values[condition]
        k_avg = np.sum(P_k*unique_degrees)
    elif net_type=='ln':
        # mu_log_norm, sigma_log_norm = -(1 / 2) * np.log((1 + epsilon ** 2) / kavg ** 2), np.sqrt(2 *np.log(kavg) +
        #                             np.log((1 + epsilon ** 2) / kavg ** 2))

        mu_log = np.log(kavg / np.sqrt(1 + epsilon ** 2))
        sigma_log = np.sqrt(np.log(1 + epsilon ** 2))

        k_values = np.linspace(1, N, N)

        pdf_values = lognorm.pdf(k_values, s=sigma_log, loc=0, scale=np.exp(mu_log))

        # pdf_values = lognorm.pdf(k_values,mu_log_norm,sigma_log_norm )  # Using loc and scale
        # Apply the condition N * pdf_values > 1
        condition = N * pdf_values > 1

        # Filter k_values and pdf_values based on the condition
        unique_degrees = k_values[condition]
        P_k = pdf_values[condition]
        k_avg = np.sum(P_k*unique_degrees)
    # elif net_type=='bet':
    #     alpha_beta_dist, beta_beta_dist = (N - kavg * (1 + epsilon ** 2)) / (N * epsilon ** 2), (
    #                 (kavg - N) * (kavg - N + kavg * epsilon ** 2)) / (kavg * N * epsilon ** 2)
    #     d = (np.random.default_rng().beta(alpha_beta_dist, beta_beta_dist, N) * N).astype(int)
    #     # k_values = np.linspace(1, N, N)
    #     # k_values = np.linspace(0, 1, N)
    #
    #     pdf_values = beta_sci.pdf(k_values, a=alpha_beta_dist,b=beta_beta_dist)
    #
    #     # Apply the condition N * pdf_values > 1
    #     condition = N * pdf_values > 1
    #
    #     # Filter k_values and pdf_values based on the condition
    #     unique_degrees = k_values[condition]
    #     P_k = pdf_values[condition]
    #     k_avg = np.sum(P_k*unique_degrees)
    elif net_type=='bd':
        unique_degrees=np.array([kavg*(1-epsilon),kavg*(1+epsilon)])
        P_k =np.array([0.5,0.5])
        k_avg= kavg

    # Initialize the matrix C
    C = np.zeros((len(unique_degrees), len(unique_degrees)))
    # Construct the matrix C
    for i, k in enumerate(unique_degrees):
        for j, k_prime in enumerate(unique_degrees):
            P_k_given_k = ((1 - correlation) * k_prime * P_k[j] / k_avg) + (correlation * (k == k_prime))
            C[i, j] = k * P_k_given_k

    # Calculate the eigenvalues of the matrix C
    eigenvalues = eigvals(C)

    # Find the largest eigenvalue
    largest_eigenvalue = np.max(eigenvalues.real)
    beta = lam/largest_eigenvalue

    x_initial = (1-1/lam)*np.ones(len(unique_degrees))
    # x_solution = fsolve(equations, x_initial)
    # Solve the equation for each degree k
    # x_solution = np.array([fsolve(equations, x_initial[j], args=(beta, k, alpha,unique_degrees))[0]
    #                        for j, k in enumerate(unique_degrees)])
    x_solution = fsolve(equations, x_initial, args=(beta, correlation,unique_degrees,k_avg))
    return x_solution,P_k


def assortative_rate_equations(vars, R0, alpha, epsilon):
    # As defined previously
    x1 = vars[0]
    x2 = vars[1]
    denom = (1 + alpha + epsilon ** 2 - alpha * epsilon ** 2 +
             np.sqrt((-1 + alpha) ** 2 - 2 * (-1 + (-2 + alpha) * alpha) * epsilon ** 2 +
                     (-1 + alpha) ** 2 * epsilon ** 4))

    eq1 = (-x1 + (R0 * (-1 + x1) * (-1 + epsilon) *
                  (-x2 * (-1 + alpha) * (1 + epsilon) +
                   x1 * (1 + alpha + (-1 + alpha) * epsilon))) / denom)

    eq2 = (-x2 - (R0 * (-1 + x2) * (1 + epsilon) *
                  (x1 * (-1 + alpha) * (-1 + epsilon) +
                   x2 * (1 + alpha + epsilon - alpha * epsilon))) / denom)

    return [eq1, eq2]


def equations(x,beta,correlation,unique_degrees,k_avg):
    eqs = []
    for j, k in enumerate(unique_degrees):
        phi_x = Phi(x,k_avg)
        eq = beta * k * (1 - x[j]) * (correlation * x[j] + (1 - correlation) * phi_x) - x[j]
        eqs.append(eq)
    return np.array(eqs)

def numerical_endemic_state_from_rate_eq(R0, epsilon, alpha):
    # As defined previously
    initial_guess = [1 - 1 / R0, 1 - 1 / R0]
    solution = fsolve(lambda vars: assortative_rate_equations(vars, R0, alpha, epsilon), initial_guess)
    solution = solution / 2
    return solution[0] + solution[1]


def assortative_network_infected_fraction(R0, alpha, epsilon):
    # As defined previously
    x = np.zeros((len(epsilon), len(alpha)))
    for i in range(len(epsilon)):
        for j in range(len(alpha)):
            x[i, j] = numerical_endemic_state_from_rate_eq(R0, epsilon[i], alpha[j])
    return x


def find_closest_column(x, I):
    # As defined previously
    differences = np.abs(x - I)
    closest_indices = np.argmin(differences, axis=1)
    return closest_indices


def find_closest_row(x, I):
    """
    Finds the row index in each column of matrix x where the value is closest to the scalar I.

    Parameters:
        x (np.ndarray): 2D array (matrix) of values.
        I (float): Scalar value to find the closest value to in each column.

    Returns:
        np.ndarray: Array of indices representing the row in each column closest to I.
    """
    # Calculate the absolute difference between each element in x and I
    differences = np.abs(x - I)

    # Find the index of the minimum difference in each column
    closest_indices = np.argmin(differences, axis=0)

    return closest_indices

def find_constant_infected_fraction(R0, alpha, epsilon,fraction,kavg_mean, degree_dist_type, N_mean):
    """
    Finds epsilon values that maintain a constant infected fraction given alpha.

    Parameters:
        R0 (float): Basic reproduction number.
        alpha (np.ndarray): Array of correlation values.
        epsilon (np.ndarray): Array of coefficient variation values.
        x (np.ndarray): Matrix of infected fractions for each epsilon-alpha pair.
        x_isomte (float): Target infected fraction value.

    Returns:
        tuple: (alpha_filtered, epsilon_filtered), filtered alpha and epsilon values.
    """
    # Find the closest row index for each row to the target infected fraction
    x0 =1-1/R0
    x_isomte = x0*fraction
    # beta = R0/(k0*(1+epsilon**2))
    # x= assortative_network_infected_fraction(R0,alpha,epsilon)
    x_star = np.array(numerical_xstar_change_eps_alpha(kavg_mean, epsilon, alpha, degree_dist_type, N_mean, R0))

    # x_initial = (1 - 1 / R0) * np.ones(len(unique_degrees))
    # x_solution = fsolve(equations, x_initial, args=(beta, correlation, unique_degrees, k0))
    # index_of_eps = find_closest_row(x, x_isomte)
    index_of_eps = find_closest_row(x_star, x_isomte)


    # Get number of rows in x
    # num_rows = x.shape[0]
    num_rows = x_star.shape[0]

    # Filter out edge values
    mask = (index_of_eps != num_rows - 1) & (index_of_eps != 0)
    filtered_indices = index_of_eps[mask]
    alpha_filtered = alpha[mask]
    epsilon_filtered = epsilon[filtered_indices]

    return alpha_filtered, epsilon_filtered

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process network and WE method parameters.")

    # Parameters for the network
    parser.add_argument('--N', type=int, help='Number of nodes')
    parser.add_argument('--prog', type=str, help='Program')
    parser.add_argument('--eps_din', type=float, help='The normalized std (second moment divided by the first) of the in-degree distribution')
    parser.add_argument('--correlation', type=float, help='Correlation parameter')
    parser.add_argument('--k', type=int, help='Average number of neighbors for each node')
    parser.add_argument('--lam', type=float, help='The reproduction number')
    parser.add_argument('--fraction', type=float, help='The fraction of the infected population to conserve')
    parser.add_argument('--num_points_alpha_eps_xstar', type=float, help='The number of points that each ')

    args = parser.parse_args()

    N = 5000 if args.N is None else args.N
    prog = 'bet' if args.prog is None else args.prog
    lam = 1.3 if args.lam is None else args.lam
    eps_din = 0.5 if args.eps_din is None else args.eps_din
    correlation = 0.3 if args.correlation is None else args.correlation
    k = 50 if args.k is None else args.k
    fraction = 0.9 if args.fraction is None else args.fraction
    num_points_alpha_eps_xstar = 20 if args.num_points_alpha_eps_xstar is None else args.num_points_alpha_eps_xstar
    alpha,epsilon = np.linspace(-correlation,correlation,num_points_alpha_eps_xstar),np.linspace(0.3,eps_din,num_points_alpha_eps_xstar)
    # lam, alpha, epsilon, fraction = 1.3,np.linspace(10**-6,1.0,num_points_alpha_eps_xstar),np.linspace(10**-6,1.0,num_points_alpha_eps_xstar),0.9
    alpha_filtered, epsilon_filtered = find_constant_infected_fraction(lam, alpha, epsilon, fraction,k,prog,N)

