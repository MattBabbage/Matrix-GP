# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas
import numpy as np
import json

def generate_matrix_data(n, matrix_size, min=0, max=1):
    df = []
    for i in range(0, n):
        a = np.random.randint(10, size=(matrix_size, matrix_size))
        b = np.random.randint(10, size=(matrix_size, matrix_size))
        c = np.matmul(a, b)
        df.append([a, b, c])
        np.save('matrices.npy', df)


    #np.savetxt("matrices.npy", df, header="a,b,c", delimiter=",", comments='')

def read_matrices(matrix):
    for i in matrix:
        for j in i:
            print("m:", j)

generate_matrix_data(20,2,0,5);
z = np.load('matrices.npy')
read_matrices(z)

