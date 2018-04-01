import math
import numpy as np
import scipy.linalg
import scipy.cluster
import scipy.sparse
import scipy.sparse.csgraph

# Controls weight of geodesic to angular distance. Values closer to 0 give
# the angular distance more importance, values closer to 1 give the geodesic
# distance more importance
delta = 0.03

# Weight of convexity. Values close to zero give more importance to concave
# angles, values close to 1 treat convex and concave angles more equally
eta = 0.15

k = 10

def processfile(fname):
	# Read an obj file and return its faces and vertices
	with open(fname) as f:
		faces_array = []
		vertices_array = []
		adjacency_map = {}
		num_faces = 0
		content = f.readlines()
		for iterator in content:
			if len(iterator) > 0:
				if iterator[0] == 'f':
					split_values = iterator[1:].split()
					face = np.asfarray(split_values,float)
					faces_array.append(face)
					edge1 = (face[0], face[1])
					edge1_back = edge1[::-1]
					edge2 = (face[1], face[2])
					edge2_back = edge2[::-1]
					edge3 = (face[2], face[0])
					edge3_back = edge3[::-1]
					if edge1_back not in adjacency_map:
						adjacency_map[edge1] = [num_faces]
					else:
						adjacency_map[edge1_back].append(num_faces)
					if edge2_back not in adjacency_map:
						adjacency_map[edge2] = [num_faces]
					else:
						adjacency_map[edge2_back].append(num_faces)
					if edge3_back not in adjacency_map:
						adjacency_map[edge3] = [num_faces]
					else:
						adjacency_map[edge3_back].append(num_faces)
					num_faces += 1
					#print faces
				elif iterator[0] == 'v':
					split_values = iterator[1:].split()
					vertex = np.asfarray(split_values,float)
					vertices_array.append(vertex)
					#print vertices

	return faces_array, vertices_array, adjacency_map

def geodesic_distance(vertices_array, face1, face2, edge):
	# Computes the geodesic distance over the given edge between
	# the two adjacent faces face1 and face2
	edge_center = (vertices_array[int(edge[0]) - 1] + vertices_array[int(edge[1]) - 1])/2
	return np.linalg.norm(edge_center - face_center(vertices_array, face1)) + np.linalg.norm(edge_center - face_center(vertices_array, face2))

def face_normal(vertices_array, face):
	# Computes the normal to a given face
	normal_nonunit = np.cross(vertices_array[int(face[1]) - 1] - vertices_array[int(face[0]) - 1], vertices_array[int(face[2]) - 1] - vertices_array[int(face[1]) - 1])
	return normal_nonunit/np.linalg.norm(normal_nonunit)

def face_center(vertices_array, face): #face is simply a 3-element array
	# Computes the coordinates of the center of the given face
	center = [0.0, 0.0, 0.0]
	for vert in face:
		center += vertices_array[int(vert) - 1]
	return center/3

def angular_distance(vertices_array, face1, face2):
	# Computes the angular distance of the given adjacent faces
	face1_normal = face_normal(vertices_array, face1)
	vector_joining_centers = face_center(vertices_array, face2) - face_center(vertices_array, face1)
	use_eta = np.dot(face1_normal, vector_joining_centers) < 0
	return use_eta , 1 - np.dot(face_normal(vertices_array, face1) , face_normal(vertices_array, face2))/(np.linalg.norm(face_normal(vertices_array, face1)) * np.linalg.norm(face_normal(vertices_array, face2)))

def create_distance_matrices(faces_array, vertices_array, adjacency_map):
	# Creates the matrices of the angular and geodesic distances between all adjacent faces. 
	# The i,j-th entry of the returned matrices contains the distance between the i-th and j-th face

	faces = faces_array
	l = len(faces_array)

	# saves, which entries in A have to be scaled with eta
	use_eta_list = []

	# number of pairs of adjacent faces
	num_adj = 0

	# map from edge-key to adjacent faces
	# adj_faces_map = {}
	# find adjacent faces by iterating edges
	# for index, face in enumerate(faces):
	# 	for edge in :
	# 		if edge in adj_faces_map:
	# 			adj_faces_map[edge].append(index)
	# 		else:
	# 			adj_faces_map[edge] = [index]

	# average G and cumulated A
	avgG = 0
	sumA = 0
	# helping vectors to create sparse matrix later on
	Arow = []
	Acol = []
	Aval = []
	Grow = []
	Gcol = []
	Gval = []
	# iterate adjacent faces and calculate distances
	for edge, adj_faces in adjacency_map.items():
		if len(adj_faces) == 2:
			i = adj_faces[0]
			j = adj_faces[1]

			Gtemp = geodesic_distance(vertices_array, faces[i], faces[j], edge)
			use_eta, Atemp = angular_distance(vertices_array, faces[i], faces[j])
			Gval.append(Gtemp)
			Grow.append(i)
			Gcol.append(j)
			Gval.append(Gtemp)  # add symmetric entry
			Grow.append(j)
			Gcol.append(i)
			Aval.append(Atemp)
			Arow.append(i)
			Acol.append(j)
			Aval.append(Atemp)  # add symmetric entry
			Arow.append(j)
			Acol.append(i)

			avgG += Gtemp
			if use_eta:
				# this entry has to be scaled with eta
				use_eta_list.append((i,j))
			else:
				# doesn't need eta so add it to the sum, if we
				# need eta we have to add it to the sum later
				sumA += Atemp
			num_adj += 1

		elif len(adj_faces) > 2:
			print("Edge with more than 2 adjacent faces: " + str(adj_faces) + "!")

	# create sparse matrices
	# matrix of geodesic distances
	G = scipy.sparse.csr_matrix((Gval, (Grow, Gcol)), shape=(l, l))
	# matrix of angular distances
	A = scipy.sparse.csr_matrix((Aval, (Arow, Acol)), shape=(l, l))

	avgG /= num_adj

	return G, A, avgG, sumA, num_adj, use_eta_list

def create_affinity_matrix(faces_array, vertices_array, adjacency_map):
	# Create the adjacency matrix of the given mesh
	l = len(faces_array)
	print("mesh_segmentation: Creating distance matrices...")
	G, A, avgG, sumA, num_adj, use_eta_list = create_distance_matrices(faces_array, vertices_array, adjacency_map)

	# scale needed angular distances with eta
	for indices in use_eta_list:
		A[indices[0], indices[1]] *= eta
		A[indices[1], indices[0]] *= eta
		sumA += A[indices[0], indices[1]]
	avgA = sumA/num_adj

	# weight with delta and average value
	G = G.dot(delta/avgG)
	A = A.dot((1 - delta)/avgA)

	print("mesh_segmentation: Finding shortest paths between all faces...")
	# for each non adjacent pair of faces find shortest path of adjacent faces
	W = scipy.sparse.csgraph.dijkstra(G + A, directed = False)
	inf_indices = np.where(np.isinf(W))
	W[inf_indices] = 0

	print("mesh_segmentation: Creating affinity matrix...")
	# change distance entries to similarities
	sigma = W.sum()/(l ** 2)
	den = 2 * (sigma ** 2)
	W = np.exp(-W/den)
	W[inf_indices] = 0
	np.fill_diagonal(W, 1)

	return W

def initial_guess(Q, k):
	# Computes an initial guess for the cluster-centers
	n = Q.shape[0]
	min_value = 2
	min_indices=(-1,-1)
	for (i,j), value in np.ndenumerate(Q):
		if i != j and value < min_value:
			min_value = Q[i,j]
			min_indices = (i,j)
	chosen = [min_indices[0], min_indices[1]]
	for _ in range(2,k):
		min_max = float("inf")
		cur_max = 0
		new_index = -1
		for i in range(n):
			if i not in chosen:
				cur_max = Q[chosen,i].max()
				if cur_max < min_max:
					min_max = cur_max
					new_index = i
		chosen.append(new_index)
	return chosen
	
filename = "data/bunny5k.obj"
faces_array, vertices_array, adjacency_map = processfile(filename) 

# affinity matrix
W = create_affinity_matrix(faces_array, vertices_array, adjacency_map)
print("mesh_segmentation: Calculating graph laplacian...")
# degree matrix
Dsqrt = np.diag([math.sqrt(1/entry) for entry in W.sum(1)])
# graph laplacian
L = Dsqrt.dot(W.dot(Dsqrt))

print("mesh_segmentation: Calculating eigenvectors...")
# get eigenvectors
l,V = scipy.linalg.eigh(L, eigvals = (L.shape[0] - k, L.shape[0] - 1))
# normalize each column to unit length
V = V / [np.linalg.norm(column) for column in V.transpose()]

print("mesh_segmentation: Preparing kmeans...")
# compute association matrix
Q = V.dot(V.transpose())
# compute initial guess for clustering
initial_clusters = initial_guess(Q, k)

print("mesh_segmentation: Applying kmeans...")
# apply kmeans
cluster_res,_ = scipy.cluster.vq.kmeans(V, V[initial_clusters,:])
# get identification vector
idx,_ = scipy.cluster.vq.vq(V, cluster_res)

print("mesh_segmentation: Done clustering!")