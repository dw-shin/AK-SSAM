import numpy as np
from fem_2d_triangle import *

iter = 6
xl, xr, yl, yr=0, 1, 0, 1
M = 2 ** np.arange(2,iter+2)
f = lambda x: 2 * np.pi**2 * np.sin(np.pi * x[:,0]) * np.sin(np.pi * x[:,1])
u_D = lambda x: 0 * x[:,0]
ux = lambda x: np.pi * np.cos(np.pi * x[:,0]) * np.sin(np.pi * x[:,1])
uy = lambda x: np.pi * np.sin(np.pi * x[:,0]) * np.cos(np.pi * x[:,1])

h = 1 / M
k = 2
error = np.zeros(iter)
M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R = get_matrices_2d_triangle(k)
for i in range(iter):
	c4n, n4e, n4db, ind4e = mesh_fem_2d(xl, xr, yl, yr, M[i], M[i], k)
	u = fem_for_poisson_2d(c4n,n4e,n4db,ind4e,M_R,Srr_R,Srs_R,Ssr_R,Sss_R,f,u_D)
	error[i] = compute_error_fem_2d(c4n,n4e,ind4e,M_R,Dr_R,Ds_R,u,ux,uy)

rate = (np.log(error[1:])-np.log(error[:-1]))/(np.log(h[1:])-np.log(h[:-1]))
print(rate)