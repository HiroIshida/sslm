from sslm import *
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import minimize
import time 

def show2d(func, bmin, bmax, N = 20, fax = None, levels = None):
    # fax is (fix, ax)
    # func: np.array([x1, x2]) list -> scalar
    # func cbar specifies the same height curve
    if fax is None:
        fig, ax = plt.subplots() 
    else:
        fig = fax[0]
        ax = fax[1]

    mat_contour_ = np.zeros((N, N))
    mat_contourf_ = np.zeros((N, N))
    x1_lin, x2_lin = [np.linspace(bmin[i], bmax[i], N) for i in range(bmin.size)]
    for i in range(N):
        for j in range(N):
            x = np.array([x1_lin[i], x2_lin[j]])
            val_c, val_cf = func(x)
            mat_contour_[i, j] = val_c
            mat_contourf_[i, j] = val_cf
    mat_contour = mat_contour_.T
    mat_contourf = mat_contourf_.T
    X, Y = np.meshgrid(x1_lin, x2_lin)

    cs = ax.contour(X, Y, mat_contour, levels = levels, cmap = 'jet')
    zc = cs.collections[0]
    plt.setp(zc, linewidth=4)
    ax.clabel(cs, fontsize=10)
    cf = ax.contourf(X, Y, mat_contourf, cmap = 'gray_r')
    fig.colorbar(cf)

def modified_banana_function(x_):
    a = 1.0
    b = 10.0
    x = ((x_ - np.array([0, -0.5])) * 4)
    if x[0] > 0:
        x[0] = -x[0] 

    binary = (a - x[0])**2 + b*(x[1]-x[0]**2)**2 - 10 < 0
    return binary

def gen_dataset(N):
    predicate = lambda x: x[0]**2 + x[1]**2 < 0.5 ** 2
    #predicate = lambda x: modified_banana_function(x)

    xp_lst = []
    xm_lst = []
    for i in range(N):
        x = rn.random(2) * 2 + np.array([-1, -1])
        if predicate(x): 
            xp_lst.append(x)
        else:
            xm_lst.append(x)
    n_p = len(xp_lst)
    yp_lst = [+1 for i in range(len(xp_lst))]
    ym_lst = [-1 for i in range(len(xm_lst))]
    x_lst = xp_lst + xm_lst
    y_lst = yp_lst + ym_lst

    X = np.zeros((2, N))
    for i in range(N):
        X[:,i] = x_lst[i]
    Y = np.array(y_lst)
    return X, Y, n_p


if __name__ == '__main__':
    rn.seed(5)
    N = 1000
    X, y, n_p = gen_dataset(N)
    rbf_kernel_custom = lambda *args: rbf_kernel(*args, gamma = 10.0)
    #kern = linear_kernel
    kern = rbf_kernel_custom

    sslm = SSLM(X, y, kern, nu = 1.0, nu1 = 0.01, nu2 = 0.01)
    sslm.predict([0, 0.8])


    t_pos = (sslm.m1 + 1.0)/(sslm.m1 + 2.0)
    t_neg = 1.0/(sslm.m2 + 2.0)
    logical = (y > 0)
    t_vec = logical * t_pos + ~logical * t_neg
    f_vec = np.array([sslm.predict(sslm.X[:, i]) for i in range(N)])

    def F(A, B):
        probs = 1.0/(1.0 + np.exp(A * f_vec + B))
        values = - (t_vec * np.log(probs) + (1 - t_vec) * np.log(1 - probs))
        val = sum(values)
        return val

    def F_(x):
        return F(x[0], x[1])

    ts = time.time()
    sol = minimize(F_, [0.1, 0.1], method = "powell")
    te = time.time()
    t_diff = te - ts
    A = sol.x[0]; B = sol.x[1]
    prob = lambda x: 1.0/(1 + exp(A * sslm.predict(x) + B))
    print(A)
    print(B)
    print(t_diff)


    
    ## plot
    """
    fig, ax = plt.subplots() 
    bmin = np.array([-1, -1])
    bmax = np.array([1, 1])
    def f(x):
        tmp = prob(x)
        return tmp, tmp
    show2d(f, bmin, bmax, fax = (fig, ax))

    idx_positive = np.where(y == 1)
    idx_negative = np.where(y == -1)

    plt.scatter(X[0, idx_positive], X[1, idx_positive], c = "blue", s = 10)
    plt.scatter(X[0, idx_negative], X[1, idx_negative], c = "red", s = 10)

    plt.scatter(X[0, sslm.idxes_SVp], X[1, sslm.idxes_SVp], c = "blue", marker = 's', s = 30)
    plt.scatter(X[0, sslm.idxes_SVn], X[1, sslm.idxes_SVn], c = "red", marker = 's', s = 30)
    plt.show()
    """


