import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def mesh_fem_1d(a,b,M,k):
	"""
	mesh_fem_1d(a, b, M, k) generates an uniform mesh on the domain [a,b] 
	in 1D with mesh size h = 1/M. Also this code returns an index matrix 
	for continuous k-th order polynomial approximations.

	Parameters:
	  - a : left-end point for the domain
	  - b : right-end point for the domain
	  - M : the number of elements
	  - k : polynomial order for the approximate solution
%
	Returns
	  - c4n : coordinates for All nodes.
	  - n4e : nodes for elements.
	  - n4db : nodes for Dirichlet boundary.
	  - ind4e : index for elements
	"""
	nrNodes = k*M + 1
	c4n = np.linspace(a, b, nrNodes)
	n4e = np.array([[i*k, (i+1)*k] for i in range(M)])
	n4db = np.array([0, nrNodes-1])
	ind4e = np.array([list(range(i*k, (i+1)*k+1)) for i in range(M)])
	return (c4n,n4e,n4db,ind4e)


def get_matrices_1d(k=1):
	"""
	get_matrices_1d(k) generates the mass matrix M_R, the stiffness
	matrix S_R and the differentiation matrix D_R for continuous k-th 
	order polynomial approximations on the reference interval.

	Paramters
		- ``k`` (``int32``) : polynomial order for the approximate solution
	
	Returns
		- ``M_R`` (``float64 array``) : Mass matrix on the reference interval
		- ``S_R`` (``float64 array``) : Stiffness matrix on the reference interval
		- ``D_R`` (``float64 array``) : Differentiation matrix on the reference interval

	Example
		>>> k = 1
		>>> M_R, S_R, D_R = get_matrices_1d(1)
		>>> M_R
		array([[0.66666667, 0.33333333],
       		   [0.33333333, 0.66666667]])
		>>> S_R
		array([[ 0.5, -0.5],
       		   [-0.5,  0.5]])
		>>> D_R
		array([[-0.5,  0.5],
       		   [-0.5,  0.5]])
	"""
	if k == 1:
		M_R = np.array([[2, 1],[1, 2]], dtype=np.float64) / 3.
		S_R = np.array([[1, -1],[-1, 1]], dtype=np.float64) / 2.
		D_R = np.array([[-1, 1],[-1, 1]], dtype=np.float64) / 2.
	elif k == 2:
		M_R = np.array([[4, 2, -1],[2, 16, 2],[-1, 2, 4]], dtype=np.float64) / 15.
		S_R = np.array([[7, -8, 1],[-8, 16, -8],[1, -8, 7]], dtype=np.float64) / 6.
		D_R = np.array([[-3, 4, -1],[-1, 0, 1],[1, -4, 3]], dtype=np.float64) / 2.

	###################################################################################################
	# TODO: Add the matrices for the cubic approximations ($k=3$).
	
	###################################################################################################	
	else:
		M_R = 0
		S_R = 0
		D_R = 0
	return (M_R, S_R, D_R)


def fem_for_poisson_1d(c4n,n4e,n4db,ind4e,k,M_R,S_R,f,u_D):
	number_of_nodes = len(c4n)
	number_of_elements = len(n4e)
	b = np.zeros(number_of_nodes)
	u = np.zeros(number_of_nodes)
	J = np.array([(c4n[n4e[i,1]]-c4n[n4e[i,0]])/2 for i in range(number_of_elements)])
	Aloc = np.array([S_R.flatten()/J[i] for i in range(number_of_elements)])
	for i in range(number_of_elements):
		b[ind4e[i]] += J[i]*np.matmul(M_R, f(c4n[ind4e[i]]))
	row_ind = np.tile(ind4e.flatten(),(k+1,1)).T.flatten()
	col_ind = np.tile(ind4e,(1,k+1)).flatten()
	A_COO = coo_matrix((Aloc.flatten(), (row_ind, col_ind)), shape=(number_of_nodes, number_of_nodes))
	A = A_COO.tocsr()
	dof = np.setdiff1d(range(0,number_of_nodes), n4db)
	u[dof] = spsolve(A[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	return u


def compute_error_fem_1d(c4n,ind4e,M_R,D_R,u,Du):
	error = 0
	number_of_elements = len(ind4e)
	J = np.array([(c4n[ind4e[i,-1]]-c4n[ind4e[i,0]])/2 for i in range(number_of_elements)])
	for i in range(number_of_elements):
		u_x = np.matmul(D_R, u[ind4e[i]]) / J[i]
		D_e = Du(c4n[ind4e[i]]) - u_x
		error += J[i] * np.matmul(D_e,np.matmul(M_R,D_e))
	return np.sqrt(error)



# Exercise 2
def fem_for_poisson_1d_ex2(c4n,n4e,n4db,ind4e,k,M_R,S_R,f,u_D):
	number_of_nodes = len(c4n)
	number_of_elements = len(n4e)
	b = np.zeros(number_of_nodes)
	u = np.zeros(number_of_nodes)
	J = np.array([(c4n[n4e[i,1]]-c4n[n4e[i,0]])/2 for i in range(number_of_elements)])
	Aloc = np.array([S_R.flatten()/J[i] for i in range(number_of_elements)])
	for i in range(number_of_elements):
		b[ind4e[i]] += J[i]*np.matmul(M_R, f(c4n[ind4e[i]]))
	row_ind = np.tile(ind4e.flatten(),(k+1,1)).T.flatten()
	col_ind = np.tile(ind4e,(1,k+1)).flatten()
	A_COO = coo_matrix((Aloc.flatten(), (row_ind, col_ind)), shape=(number_of_nodes, number_of_nodes))
	A = A_COO.tocsr()
	dof = np.setdiff1d(range(0,number_of_nodes), n4db)

	###################################################################################################
	# TODO: Add a few lines to treat non-homogeneous Dirichlet boundary condition.
	
	###################################################################################################

	u[dof] = spsolve(A[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	return u


# Exercise 3
def fem_for_poisson_1d_ex3(c4n,n4e,n4db,n4nb,ind4e,k,M_R,S_R,f,u_D,u_N):
	number_of_nodes = len(c4n)
	number_of_elements = len(n4e)
	b = np.zeros(number_of_nodes)
	u = np.zeros(number_of_nodes)
	J = np.array([(c4n[n4e[i,1]]-c4n[n4e[i,0]])/2 for i in range(number_of_elements)])
	Aloc = np.array([S_R.flatten()/J[i] for i in range(number_of_elements)])
	for i in range(number_of_elements):
		b[ind4e[i]] += J[i]*np.matmul(M_R, f(c4n[ind4e[i]]))
	row_ind = np.tile(ind4e.flatten(),(k+1,1)).T.flatten()
	col_ind = np.tile(ind4e,(1,k+1)).flatten()
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