import math
import numpy as np

# Controls weight of geodesic to angular distance. Values closer to 0 give
# the angular distance more importance, values closer to 1 give the geodesic
# distance more importance
delta = None

# Weight of convexity. Values close to zero give more importance to concave
# angles, values close to 1 treat convex and concave angles more equally
eta = None

def processfile(fname):
	# Read an obj file and return its faces and vertices
	with open(fname) as f:
		faces_array = []
		vertices_array = []
		content = f.readlines()
		for iterator in content:
			if len(iterator) > 0:
				if iterator[0] == 'f':
					split_values = iterator[1:].split()
					face = np.asfarray(split_values,float)
					faces_array.append(face)
					#print faces
				elif iterator[0] == 'v':
					split_values = iterator[1:].split()
					vertex = np.asfarray(split_values,float)
					vertices_array.append(vertex)
					#print vertices

	return faces_array, vertices_array

def geodesic_distance(vertices_array, face1, face2, edge):
	# Computes the geodesic distance over the given edge between
	# the two adjacent faces face1 and face2
	edge_center = (vertices_array[edge[0]] + vertices_array[edge[1]])/2
	return np.linalg.norm(edge_center - face_center(vertices_array, face1)) + np.linalg.norm(edge_center - face_center(vertices_array, face2))

def face_normal(vertices_array, face):
	# Computes the normal to a given face
	normal_nonunit = np.cross(vertices_array[int(face[1])] - vertices_array[int(face[0])], vertices_array[int(face[2])] - vertices_array[int(face[1])])
	return normal_nonunit/np.linalg.norm(normal_nonunit)

def face_center(vertices_array, face): #face is simply a 3-element array
	# Computes the coordinates of the center of the given face
	center = [0.0, 0.0, 0.0]
	for vert in face:
		center += vertices_array[int(vert)]
	return center/3

def angular_distance(vertices_array, face1, face2):
	# Computes the angular distance of the given adjacent faces
	face1_normal = face_normal(vertices_array, face1)
	vector_joining_centers = face_center(vertices_array, face2) - face_center(vertices_array, face1)
	use_eta = np.dot(face1_normal, vector_joining_centers) < 0
	return use_eta , 1 - np.dot(face_normal(vertices_array, face1) , face_normal(vertices_array, face2))/(np.linalg.norm(face_normal(vertices_array, face1)) * np.linalg.norm(face_normal(vertices_array, face2)))

filename = "data/bunny5k.obj"
faces_array, vertices_array = processfile(filename) 
dist = geodesic_distance(vertices_array, faces_array[1], faces_array[2], [1 , 2])
print dist