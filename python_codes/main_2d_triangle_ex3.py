import numpy as np
from fem_2d_triangle import *

iter = 6
xl, xr, yl, yr=0, 1, 0, 1
M = 2 ** np.arange(2,iter+2)
f = lambda x: 2 * np.pi**2 * np.sin(np.pi * x[:,0]) * np.sin(np.pi * x[:,1])
u_D = lambda x: x[:,0]
u_N = lambda x: -np.pi * np.sin(np.pi * x[:,0]) * np.cos(np.pi * x[:,1])
ux = lambda x: np.pi * np.cos(np.pi * x[:,0]) * np.sin(np.pi * x[:,1]) + 1
uy = lambda x: np.pi * np.sin(np.pi * x[:,0]) * np.cos(np.pi * x[:,1])

h = 1 / M
k = 2
error = np.zeros(iter)
M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R = get_matrices_2d_triangle(k)
from fem_1d import get_matrices_1d
M_R1D,_,_ = get_matrices_1d(k)
for i in range(iter):
	c4n, n4e, n4db, ind4e = mesh_fem_2d(xl, xr, yl, yr, M[i], M[i], k)
	ind4nb = np.array([list(np.arange(j*k,(j+1)*k+1)) for j in range(0,M[i])])
	n4db = np.concatenate((np.array([0,k*M[i]]), np.setdiff1d(n4db,ind4nb.flatten())),axis=0)
	u = fem_for_poisson_2d_ex3(c4n,n4e,n4db,ind4nb,ind4e,M_R,Srr_R,Srs_R,Ssr_R,Sss_R,M_R1D,f,u_D,u_N)
	error[i] = compute_error_fem_2d(c4n,n4e,ind4e,M_R,Dr_R,Ds_R,u,ux,uy)

rate = (np.log(error[1:])-np.log(error[:-1]))/(np.log(h[1:])-np.log(h[:-1]))
print(rate)