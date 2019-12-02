from sklearn.metrics.pairwise import *
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import cvxopt
import math

def gen_diadig(vec):
    n = vec.size
    Y = np.array([vec])
    tmp = np.repeat(Y, n, 0)
    ymat = tmp * tmp.T
    return ymat

class SSLM:

    def __init__(self, X, y, kern, nu = 1.0, nu1 = 0.2, nu2 = 0.2):
        self.X = X
        self.y = y
        self.kern = kern
        self.nu = nu
        self.nu1 = nu1 
        self.nu2 = nu2

        self.N = len(y)
        self.m1 = (self.N + sum(y))/2
        self.m2 = self.N - self.m1


        self.R, self.rho, self.cc, self.a_lst, self.idxes_SVp, self.idxes_SVn\
                = self._compute_important_parameters()

    def predict(self, x):
        val = (self.R**2 - self.cc - self.kern(x, x, gamma = 10) + sum(2 * self.kern(self.X.T, x, gamma = 10).flatten() * self.a_lst * self.y)).item()
        return val

    def _compute_important_parameters(self):
        gram, gramymat, a_lst = self._solve_qp()

        eps = 0.00001
        idxes_S1 = []
        for i in range(self.m1):
            a = a_lst[i]
            if eps < a and a < 1.0/(self.nu1 * self.m1) - eps:
                idxes_S1.append(i)

        idxes_S2 = []
        for i in range(self.m1, self.m1 + self.m2):
            a = a_lst[i]
            if eps < a and a < 1.0/(self.nu2 * self.m2) - eps:
                idxes_S2.append(i)
        print(idxes_S1)
        print(idxes_S2)

        cc = sum(sum(gen_diadig(a_lst) * gramymat))
        f_inner = lambda idx: gram[idx, idx] - sum(2 * gram[idx, :] * a_lst * self.y) + cc

        P1 = sum(map(f_inner, idxes_S1))
        P2 = sum(map(f_inner, idxes_S2))

        n1 = len(idxes_S1)
        n2 = len(idxes_S2)
        print P1
        print P2
        R = math.sqrt(P1/n1)
        rho = math.sqrt(P2/n2 - P1/n1)
        return R, rho, cc, a_lst, idxes_S1, idxes_S2

    def _solve_qp(self):
        Ymat = gen_diadig(np.array(self.y))

        gram = self.kern(self.X.T, gamma = 10)
        gram_diag_matrix = np.diag(gram)

        gramymat = gram * Ymat
        gramymat_diag = np.array([-gram_diag_matrix]).T * np.array([self.y]).T

        P = cvxopt.matrix(gramymat)
        q = cvxopt.matrix(gramymat_diag)

        # eq 15
        A_15 = np.array([self.y],dtype = np.float64)
        b_15 = np.eye(1)
        A_16 = np.ones((1, self.N))
        b_16 = np.eye(1)*(2 * self.nu + 1)
        A_ = np.vstack((A_15, A_16))
        B_ = np.vstack((b_15, b_16))
        A = cvxopt.matrix(A_)
        b = cvxopt.matrix(B_)

        G0 = np.eye(self.N)
        G1 = - np.eye(self.N)
        G_ = np.vstack((G0, G1))
        G = cvxopt.matrix(G_)

        h0p = np.ones(self.m1)/(self.nu1 * self.m1)
        h0m = np.ones(self.N - self.m1)/(self.nu2 * (self.N - self.m1))
        h1 = np.zeros(self.N)
        h_ = np.block([h0p, h0m, h1])
        h = cvxopt.matrix(h_)
        sol = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h)
        a_lst = np.array([sol["x"][i] for i in range(self.N)])
        return gram, gramymat, a_lst

