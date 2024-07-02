import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def mesh_fem_2d(xl, xr, yl, yr, Mx, My, k):
	"""
	mesh_fem_2d_triangle    Mesh geometry on 2D rectangular domain
	   mesh_fem_2d_triangle(xl, xr, yl, yr, Mx, My, k) generates an uniform
	   triangular mesh on the domain [xl,xr]x[yl,yr] in 2D with Mx elements
	   along x-direction and My elements along y-direction. Also this code
	   returns an index matrix for continuous k-th order polynomial 
	   approximations.
	    
	
	   Parameters:
	     - xl : x-coordinate of bottom-left vertex of the domain
	     - xr : x-coordinate of top-right vertex of the domain
	     - yl : y-coordinate of bottom-left vertex of the domain
	     - yr : y-coordinate of top-right vertex of the domain
	     - Mx : the number of elements along x-direction
	     - My : the number of elements along y-direction
	     - k : polynomial order for the approximate solution
	
	   Returns:
	     - c4n    coordinates for nodes.
	     - n4e    nodes for elements.
	     - ind4e  indices for elements
	     - n4db   nodes for Dirichlet boundary.
	"""

	tmp = np.array([[np.arange(0,k*Mx,k) + (k*Mx+1)*i*k] for i in range(My)]).flatten()
	tmp2 = np.array([[np.arange(k,k*Mx+1,k) + (k*Mx+1)*(i+1)*k] for i in range(My)]).flatten()
	tmp3 = list(np.concatenate([list(j*(Mx*k+1)+np.arange(0,k+1-j)) for j in range(k+1)]))
	ind4e = np.array([tmp[np.int32(i/2)]+tmp3 if i % 2 == 0 else tmp2[np.int32((i-1)/2)] - tmp3 for i in range(2*Mx*My)])
	n4e = np.array([[ind4e[i,k], ind4e[i,np.int32((k+1)*(k+2)/2-1)], ind4e[i,0]] for i in range(ind4e.shape[0])])
	n4db = np.concatenate([[i for i in range(0,k*Mx+1)], [i for i in range(2*k*Mx+1,(k*Mx+1)*(k*My+1),(k*Mx+1))],
			[i for i in range((k*Mx+1)*(k*My+1)-2,k*My*(k*Mx+1)-1,-1)], [i for i in range((k*My-1)*(k*Mx+1),k*Mx,-(k*Mx+1))]])
	x = np.linspace(xl,xr,k*Mx+1)
	y = np.linspace(yl,yr,k*My+1)
	x = np.tile(x, (1,k*My+1)).flatten()
	y = np.tile(y,(k*Mx+1,1)).T.flatten()
	c4n = np.array([[x[i], y[i]] for i in range(len(x))])
	return (c4n,n4e,n4db,ind4e)


def get_matrices_2d_triangle(k=1):
	"""
	get_matrices_2d_triangle    Matrices for Poisson using FEM in 2D
	   get_matrices_2d_triangle(k) generates the mass matrix M_R, the 
	   stiffness matrices Srr_R, Srs_R, Ssr_R and Sss_R, and the 
	   differentiation matrices Dr_R and Ds_R for continuous k-th order 
	   polynomial approximations on the reference interval.
	
	   Parameters:
	     - k : polynomial order for the approximate solution
	
	   Returns:
	     - M_R : Mass matrix on the reference interval
	     - Srr_R : Stiffness matrix on the reference interval
	     - Srs_R : Stiffness matrix on the reference interval
	     - Ssr_R : Stiffness matrix on the reference interval
	     - Sss_R : Stiffness matrix on the reference interval
	     - Dr_R : Differentiation matrix with respect to r on the reference 
	              triangle
	     - Ds_R : Differentiation matrix with respect to s on the reference 
	              triangle
	"""
	if k == 1:
		M_R = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])/6.
		Srr_R = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])/2.
		Srs_R = np.array([[1, 0, -1], [-1, 0, 1], [0, 0, 0]])/2.
		Ssr_R = np.array([[1, -1, 0], [0, 0, 0], [-1, 1, 0]])/2.
		Sss_R = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])/2.
		Dr_R = np.array([[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]])/2.
		Ds_R = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])/2.
	elif k == 2:
		M_R = np.array([[6, 0, -1, 0, -4, -1], [0, 32, 0, 16, 16, -4], [-1, 0, 6, -4, 0, -1], 
			[0, 16, -4, 32, 16, 0], [-4, 16, 0, 16, 32, 0], [-1, -4, -1, 0, 0, 6]])/90.
		Srr_R = np.array([[3 ,-4, 1, 0, 0, 0], [-4, 8, -4, 0, 0, 0], [1, -4, 3, 0, 0, 0], 
			[0, 0, 0, 8, -8, 0], [0, 0, 0, -8, 8, 0], [0, 0, 0, 0, 0, 0]])/6.
		Srs_R = np.array([[3, 0, 0, -4, 0, 1], [-4, 4, 0, 4, -4, 0], [1, -4, 0, 0, 4, -1], 
			[0, 4, 0, 4, -4, -4], [0, -4, 0, -4, 4, 4], [0, 0, 0, 0, 0, 0]])/6.
		Ssr_R = np.array([[3, -4, 1, 0, 0, 0], [0, 4, -4, 4, -4, 0], [0, 0, 0, 0, 0, 0], 
			[-4, 4, 0, 4, -4, 0], [0, -4, 4, -4, 4, 0], [1, 0, -1, -4, 4, 0]])/6.
		Sss_R = np.array([[3, 0, 0, -4, 0, 1], [0, 8, 0, 0, -8, 0], [0, 0, 0, 0, 0, 0], 
			[-4, 0, 0, 8, 0, -4], [0, -8, 0, 0, 8, 0], [1, 0, 0, -4, 0, 3]])/6.
		Dr_R = np.array([[-3, 4, -1, 0, 0, 0], [-1, 0, 1, 0, 0, 0], [1, -4, 3, 0, 0, 0], 
			[-1, 2, -1, -2, 2, 0], [1, -2, 1, -2, 2, 0], [1, 0, -1, -4, 4, 0]])/2.
		Ds_R = np.array([[-3, 0, 0, 4, 0, -1], [-1, -2, 0, 2, 2, -1], [1, -4, 0, 0, 4, -1], 
			[-1, 0, 0, 0, 0, 1], [1, -2, 0, -2, 2, 1], [1, 0, 0, -4, 0, 3]])/2.
	else:
		M_R = 0
		Srr_R = 0
		Srs_R = 0
		Ssr_R = 0
		Sss_R = 0
		Dr_R = 0
		Ds_R = 0
	return (M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R)


def fem_for_poisson_2d(c4n,n4e,n4db,ind4e,M_R,Srr_R,Srs_R,Ssr_R,Sss_R,f,u_D):
	"""
	fem_for_poisson_2d_triangle    FEM solver for Poisson problem in 2D with
	                               triangular elements
	   fem_for_poisson_2d_triangle(c4n,n4e,n4db,ind4e,M_R,Srr_R,Srs_R,Ssr_R, 
	   Sss_R,f,u_D) solves the Poisson problem. In order to use this code, 
	   mesh information (c4n,n4e,n4db,ind4e), matrices (M_R,Srr_R,Srs_R,Ssr_R, 
	   Sss_R), the source f, and the boundary condition u_D. Then the results 
	   of this code are the numerical solution u, the global stiffness matrix 
	   A, the global load vector b and the freenodes.
	
	   Parameters:
	     - c4n : coordinates for nodes.
	     - n4e : nodes for elements.
	     - n4db : nodes for Dirichlet boundary.
	     - ind4e : indices for elements
	     - M_R : Mass matrix on the reference triangle
	     - Srr_R : Stiffness matrix on the reference triangle
	     - Srs_R : Stiffness matrix on the reference triangle
	     - Ssr_R : Stiffness matrix on the reference triangle
	     - Sss_R : Stiffness matrix on the reference triangle
	     - f : RHS in the Poisson problem
	     - u_D : Dirichlet boundary condition for the solution u
	
	   Returns:
	     - u : numerical solution
	"""
	number_of_nodes = c4n.shape[0]
	number_of_elements = n4e.shape[0]
	b = np.zeros(number_of_nodes)
	u = np.zeros(number_of_nodes)
	xr = np.array([(c4n[n4e[i,0],0] - c4n[n4e[i,2],0])/2. for i in range(number_of_elements)])
	yr = np.array([(c4n[n4e[i,0],1] - c4n[n4e[i,2],1])/2. for i in range(number_of_elements)])
	xs = np.array([(c4n[n4e[i,1],0] - c4n[n4e[i,2],0])/2. for i in range(number_of_elements)])
	ys = np.array([(c4n[n4e[i,1],1] - c4n[n4e[i,2],1])/2. for i in range(number_of_elements)])
	J = xr*ys - xs*yr
	rx=ys/J
	ry=-xs/J
	sx=-yr/J
	sy=xr/J
	Aloc = np.array([J[i]*((rx[i]**2+ry[i]**2)*Srr_R.flatten() + (rx[i]*sx[i]+ry[i]*sy[i])*(Srs_R.flatten()+Ssr_R.flatten()) 
		+ (sx[i]**2+sy[i]**2)*Sss_R.flatten()) for i in range(number_of_elements)])
	for i in range(number_of_elements):
		b[ind4e[i]] += J[i]*np.matmul(M_R, f(c4n[ind4e[i]]))
	row_ind = np.tile(ind4e.flatten(),(ind4e.shape[1],1)).T.flatten()
	col_ind = np.tile(ind4e,(1,ind4e.shape[1])).flatten()
	A_COO = coo_matrix((Aloc.flatten(), (row_ind, col_ind)), shape=(number_of_nodes, number_of_nodes))
	A = A_COO.tocsr()
	dof = np.setdiff1d(range(0,number_of_nodes), n4db)
	u[dof] = spsolve(A[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	return u


def compute_error_fem_2d(c4n,n4e,ind4e,M_R,Dr_R,Ds_R,u,ux,uy):
	"""
	compute_error_fem_2d_triangle    Semi H1 error (2D triangular element)
	   compute_error_fem_2d_triangle(c4n,n4e,ind4e,M_R,Dr_R,Ds_R,u,ux,uy) 
	   computes the semi H1 error between the exact solution and the FE solution.
	
	   Parameters:
	     - c4n : coordinates for nodes.
	     - n4e : nodes for elements.
	     - ind4e : indices for elements
	     - M_R : Mass matrix on the reference triangle
	     - Dr_R : Differentiation matrix with respect to r on the reference 
	              triangle
	     - Ds_R : Differentiation matrix with respect to s on the reference 
	              triangle
	     - u : numerical solution
	     - ux : Derivative of the exact solution with respect to x for the 
	            model problem
	     - uy : Derivative of the exact solution with respect to y for the 
	            model problem
	
	   Returns:
	     - error  Semi H1 error between the exact solution and the FE solution.
	"""
	error = 0
	number_of_elements = n4e.shape[0]
	xr = np.array([(c4n[n4e[i,0],0] - c4n[n4e[i,2],0])/2. for i in range(number_of_elements)])
	yr = np.array([(c4n[n4e[i,0],1] - c4n[n4e[i,2],1])/2. for i in range(number_of_elements)])
	xs = np.array([(c4n[n4e[i,1],0] - c4n[n4e[i,2],0])/2. for i in range(number_of_elements)])
	ys = np.array([(c4n[n4e[i,1],1] - c4n[n4e[i,2],1])/2. for i in range(number_of_elements)])
	J = xr*ys-xs*yr
	rx=ys/J
	ry=-xs/J
	sx=-yr/J
	sy=xr/J
	for i in range(number_of_elements):
		Du_x = np.matmul(rx[i]*Dr_R+sx[i]*Ds_R, u[ind4e[i]])
		Du_y = np.matmul(ry[i]*Dr_R+sy[i]*Ds_R, u[ind4e[i]])
		Dex = ux(c4n[ind4e[i]]) - Du_x
		Dey = uy(c4n[ind4e[i]]) - Du_y
		error += J[i] * (np.matmul(Dex,np.matmul(M_R,Dex)) + np.matmul(Dey,np.matmul(M_R,Dey)))
	return np.sqrt(error)



# Exercise 2
def fem_for_poisson_2d_ex2(c4n,n4e,n4db,ind4e,M_R,Srr_R,Srs_R,Ssr_R,Sss_R,f,u_D):
	"""
	fem_for_poisson_2d_triangle    FEM solver for Poisson problem in 2D with
	                               triangular elements
	   fem_for_poisson_2d_triangle(c4n,n4e,n4db,ind4e,M_R,Srr_R,Srs_R,Ssr_R, 
	   Sss_R,f,u_D) solves the Poisson problem. In order to use this code, 
	   mesh information (c4n,n4e,n4db,ind4e), matrices (M_R,Srr_R,Srs_R,Ssr_R, 
	   Sss_R), the source f, and the boundary condition u_D. Then the results 
	   of this code are the numerical solution u, the global stiffness matrix 
	   A, the global load vector b and the freenodes.
	
	   Parameters:
	     - c4n : coordinates for nodes.
	     - n4e : nodes for elements.
	     - n4db : nodes for Dirichlet boundary.
	     - ind4e : indices for elements
	     - M_R : Mass matrix on the reference triangle
	     - Srr_R : Stiffness matrix on the reference triangle
	     - Srs_R : Stiffness matrix on the reference triangle
	     - Ssr_R : Stiffness matrix on the reference triangle
	     - Sss_R : Stiffness matrix on the reference triangle
	     - f : RHS in the Poisson problem
	     - u_D : Dirichlet boundary condition for the solution u
	
	   Returns:
	     - u : numerical solution
	"""
	number_of_nodes = c4n.shape[0]
	number_of_elements = n4e.shape[0]
	b = np.zeros(number_of_nodes)
	u = np.zeros(number_of_nodes)
	xr = np.array([(c4n[n4e[i,0],0] - c4n[n4e[i,2],0])/2. for i in range(number_of_elements)])
	yr = np.array([(c4n[n4e[i,0],1] - c4n[n4e[i,2],1])/2. for i in range(number_of_elements)])
	xs = np.array([(c4n[n4e[i,1],0] - c4n[n4e[i,2],0])/2. for i in range(number_of_elements)])
	ys = np.array([(c4n[n4e[i,1],1] - c4n[n4e[i,2],1])/2. for i in range(number_of_elements)])
	J = xr*ys - xs*yr
	rx=ys/J
	ry=-xs/J
	sx=-yr/J
	sy=xr/J
	Aloc = np.array([J[i]*((rx[i]**2+ry[i]**2)*Srr_R.flatten() + (rx[i]*sx[i]+ry[i]*sy[i])*(Srs_R.flatten()+Ssr_R.flatten()) 
		+ (sx[i]**2+sy[i]**2)*Sss_R.flatten()) for i in range(number_of_elements)])
	for i in range(number_of_elements):
		b[ind4e[i]] += J[i]*np.matmul(M_R, f(c4n[ind4e[i]]))
	row_ind = np.tile(ind4e.flatten(),(ind4e.shape[1],1)).T.flatten()
	col_ind = np.tile(ind4e,(1,ind4e.shape[1])).flatten()
	A_COO = coo_matrix((Aloc.flatten(), (row_ind, col_ind)), shape=(number_of_nodes, number_of_nodes))
	A = A_COO.tocsr()
	dof = np.setdiff1d(range(0,number_of_nodes), n4db)

	###################################################################################################
	# TODO: Add a few lines to treat non-homogeneous Dirichlet boundary condition. 
	#       If you finished Exercise 2, the following lines can be copied from `fem_for_poisson_1d_ex2`
	
	###################################################################################################

	u[dof] = spsolve(A[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	return u


# Exercise 3
def fem_for_poisson_2d_ex3(c4n,n4e,n4db,ind4nb,ind4e,M_R,Srr_R,Srs_R,Ssr_R,Sss_R,M_R1D,f,u_D):
	"""
	fem_for_poisson_2d_triangle    FEM solver for Poisson problem in 2D with
	                               triangular elements
	   fem_for_poisson_2d_triangle(c4n,n4e,n4db,ind4e,M_R,Srr_R,Srs_R,Ssr_R, 
	   Sss_R,f,u_D) solves the Poisson problem. In order to use this code, 
	   mesh information (c4n,n4e,n4db,ind4e), matrices (M_R,Srr_R,Srs_R,Ssr_R, 
	   Sss_R), the source f, and the boundary condition u_D. Then the results 
	   of this code are the numerical solution u, the global stiffness matrix 
	   A, the global load vector b and the freenodes.
	
	   Parameters:
	     - c4n : coordinates for nodes.
	     - n4e : nodes for elements.
	     - n4db : nodes for Dirichlet boundary.
	     - ind4e : indices for elements
	     - M_R : Mass matrix on the reference triangle
	     - Srr_R : Stiffness matrix on the reference triangle
	     - Srs_R : Stiffness matrix on the reference triangle
	     - Ssr_R : Stiffness matrix on the reference triangle
	     - Sss_R : Stiffness matrix on the reference triangle
	     - f : RHS in the Poisson problem
	     - u_D : Dirichlet boundary condition for the solution u
	
	   Returns:
	     - u : numerical solution
	"""
	number_of_nodes = c4n.shape[0]
	number_of_elements = n4e.shape[0]
	b = np.zeros(number_of_nodes)
	u = np.zeros(number_of_nodes)
	xr = np.array([(c4n[n4e[i,0],0] - c4n[n4e[i,2],0])/2. for i in range(number_of_elements)])
	yr = np.array([(c4n[n4e[i,0],1] - c4n[n4e[i,2],1])/2. for i in range(number_of_elements)])
	xs = np.array([(c4n[n4e[i,1],0] - c4n[n4e[i,2],0])/2. for i in range(number_of_elements)])
	ys = np.array([(c4n[n4e[i,1],1] - c4n[n4e[i,2],1])/2. for i in range(number_of_elements)])
	J = xr*ys - xs*yr
	rx=ys/J
	ry=-xs/J
	sx=-yr/J
	sy=xr/J
	Aloc = np.array([J[i]*((rx[i]**2+ry[i]**2)*Srr_R.flatten() + (rx[i]*sx[i]+ry[i]*sy[i])*(Srs_R.flatten()+Ssr_R.flatten()) 
		+ (sx[i]**2+sy[i]**2)*Sss_R.flatten()) for i in range(number_of_elements)])
	for i in range(number_of_elements):
		b[ind4e[i]] += J[i]*np.matmul(M_R, f(c4n[ind4e[i]]))
	row_ind = np.tile(ind4e.flatten(),(ind4e.shape[1],1)).T.flatten()
	col_ind = np.tile(ind4e,(1,ind4e.shape[1])).flatten()
	A_COO = coo_matrix((Aloc.flatten(), (row_ind, col_ind)), shape=(number_of_nodes, number_of_nodes))
	A = A_COO.tocsr()
	dof = np.setdiff1d(range(0,number_of_nodes), n4db)

	###################################################################################################
	# TODO: Add a few lines to treat Neumann boundary condition.
	
	###################################################################################################

	###################################################################################################
	# TODO: Add a few lines to treat non-homogeneous Dirichlet boundary condition. 
	#       If you finished Exercise 2, the following lines can be copied from `fem_for_poisson_1d_ex2`
	
	###################################################################################################
	
	u[dof] = spsolve(A[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	return u