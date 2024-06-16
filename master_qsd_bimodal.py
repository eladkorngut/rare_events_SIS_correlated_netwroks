import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def probability_matrix(prob_vec,L):
    prob_vec = prob_vec[:, 0].real
    prob_vec /= prob_vec.sum()
    prob_vec = np.concatenate(([0], prob_vec))

    prob_matrix = np.zeros((N // 2 + 1, N // 2 + 1))
    for col in range(L):
        I1 = col // (N // 2 + 1)
        I2 = col % (N // 2 + 1)
        prob_matrix[I1, I2] = prob_vec[col]
    return prob_matrix


def MTE(N, R0, gamma, epsilon, k0, alpha):
    k1 = k0 * (1 - epsilon)
    k2 = k0 * (1 + epsilon)
    bifurcation = 2 / (k0 * (1 + alpha + epsilon**2 - alpha * epsilon**2 +
                             np.sqrt((-1 + alpha)**2 - 2 * (-1 + (-2 + alpha) * alpha) * epsilon**2 +
                                     (-1 + alpha)**2 * epsilon**4)))
    beta = bifurcation * R0
    L = ((N // 2 + 1))**2
    i = []
    j = []
    v = []

    for col in range(2, L):
        c1 = col - 1
        c2 = (col + (N // 2) + 1) % (L + 1) - 1
        c3 = (col - (N // 2) - 1) % L - 1
        I1 = (col - 1) // (N // 2 + 1)
        I2 = (col - 1) % (N // 2 + 1)
        i.append(c1)
        j.append(col - 1)
        v.append(-(beta * (1 - alpha) * ((k1 * I1 + k2 * I2) / (k0 * N)) * (k1 * (N // 2 - I1) + k2 * (N // 2 - I2)) +
                   beta * alpha * (k1 * (2 * I1 / N) * (N // 2 - I1) + k2 * (2 * I2 / N) * (N // 2 - I2)) +
                   gamma * I1 + gamma * I2))
        if I2 + 1 <= N // 2:
            i.append(c1 + 1)
            j.append(col - 1)
            v.append(gamma * (I2 + 1))
        if I1 + 1 <= N // 2:
            i.append(c2)
            j.append(col - 1)
            v.append(gamma * (I1 + 1))
        if I1 - 1 >= 0 and I2 >= 0:
            i.append(c3)
            j.append(col - 1)
            v.append(beta * k1 * (1 - alpha) * ((k1 * (I1 - 1) + k2 * I2) / (k0 * N)) * (N // 2 - (I1 - 1)) +
                       beta * k1 * alpha * (2 * (I1 - 1) / N) * (N // 2 - (I1 - 1)))
        if I2 - 1 >= 0 and I1 >= 0:
            i.append(c1 - 1)
            j.append(col - 1)
            v.append(beta * k2 * (1 - alpha) * ((k1 * I1 + k2 * (I2 - 1)) / (k0 * N)) * (N // 2 - (I2 - 1)) +
                       beta * k2 * alpha * (2 * (I2 - 1) / N) * (N // 2 - (I2 - 1)))

    i.append(L - 1)
    j.append(L - 1)
    v.append(-(gamma * (N // 2) + gamma * (N // 2)))
    i.append(L - 2)
    j.append(L - 1)
    v.append(beta * k2 * (1 - alpha) * ((k1 * (N // 2) + k2 * (N // 2 - 1)) / (k0 * N)) * (N // 2 - (N // 2 - 1)) +
               beta * k2 * alpha * (2 * (N // 2 - 1) / N) * (N // 2 - (N // 2 - 1)))
    i.append(L - N // 2 - 2)
    j.append(L - 1)
    v.append(beta * k1 * (1 - alpha) * ((k1 * (N // 2 - 1) + k2 * (N // 2)) / (k0 * N)) * (N // 2 - (N // 2 - 1)) +
               beta * k1 * alpha * (2 * (N // 2 - 1) / N) * (N // 2 - (N // 2 - 1)))

    # Convert to high precision
    # v = np.array(v, dtype=np.float64)  # Use extended precision if supported
    # i = np.array(i, dtype=np.int64)
    # j = np.array(j, dtype=np.int64)

    Q = sp.coo_matrix((v, (i, j)), shape=(L, L)).tocsc()
    Q = Q[1:, 1:]
    Q = Q.transpose()

    # Find the largest eigenvalue
    D, prob_vec = spla.eigs(Q, k=1, which='SM')
    tau = -1 / D[0].real
    return tau

def run_multi_correlations(N,R0,gamma,epsilon,k0,alpha):
    tau=np.empty(len(alpha))
    for a in range(len(alpha)):
        tau[a] = MTE(N,R0,gamma,epsilon,k0,alpha[a])
    np.save('MTE_N_{}_R0_{}_epsilon_{}_k0_{}.npy'.format(N,R0,epsilon,k0),tau)
    parameters = np.array([N,R0,gamma,epsilon,k0])
    np.save('parameters.npy',parameters)
    np.save('alpha.npy',alpha)


if __name__ == '__main__':
    N = 1000
    R0 = 1.2
    gamma = 1.0
    epsilon = 0.5
    k0 = 100
    alpha = np.linspace(0.0,1.0,10)
    # tau = MTE(N, R0, gamma, epsilon, k0, alpha)
    run_multi_correlations(N,R0,gamma,epsilon,k0,alpha)
    # print("Extinction time (tau):", tau)
    # print("Quasi-stationary distribution matrix (prob_matrix):")
    # print(prob_matrix)
