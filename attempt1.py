import math
import numpy as np


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



def face_center(vertices_array, face): #face is simply a 3-element array
    #Computes the coordinates of the center of the given face
    center = [0.0, 0.0, 0.0]
    for vert in face:
        center += vertices_array[int(vert)]
        print int(vert), center
    return center/3

filename = "data/bunny5k.obj"
faces_array, vertices_array = processfile(filename) 
