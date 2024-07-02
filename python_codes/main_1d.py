import numpy as np
from fem_1d import *

iter = 10
a = 0
b = 1
k = 1
M = 2 ** np.arange(2,iter+2)
f = lambda x: 25 * np.pi**2 * np.sin(5 * np.pi * x)
u_D = lambda x: 0 * x
Du = lambda x: 5 * np.pi * np.cos(5 * np.pi * x)

error = np.zeros(iter)
h = 1 / M

M_R, S_R, D_R = get_matrices_1d(k)
for i in range(iter):
	c4n, n4e, n4db, ind4e = mesh_fem_1d(a,b,M[i],k)
	u = fem_for_poisson_1d(c4n,n4e,n4db,ind4e,k,M_R,S_R,f,u_D)
	error[i] = compute_error_fem_1d(c4n,ind4e,M_R,D_R,u,Du)

rate = (np.log(error[1:])-np.log(error[:-1]))/(np.log(h[1:])-np.log(h[:-1]))
print(rate)