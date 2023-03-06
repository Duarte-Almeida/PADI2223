import numpy as np
from scipy.linalg import null_space

def print_matrix(matrix):
    string = ""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            string += (str(np.round(matrix[i][j], 4)) + " & ")
        string = string[:-2] + "\\\\\n"
    print(string)


P = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [1/4, 0, 1/4, 1/4, 1/4, 0, 0],
    [0, 1/2, 0, 0, 0, 0, 1/2],
    [0, 1/2, 0, 0, 0, 1/2, 0],
    [0, 1/2, 0, 0, 0, 0, 1/2],
    [0, 0, 0, 1/2, 0, 0, 1/2],
    [0, 0, 1/3, 0, 1/3, 1/3, 0]
])

P_2 = np.linalg.matrix_power(P, 2)
#print(P_2)
#print(np.eye(1, 7, 0) @ P_2)

# stationary distribution
stat_dist = null_space((P.T - np.identity(7)))
stat_dist = stat_dist / np.sum(stat_dist)

print_matrix(stat_dist.T)
print_matrix(stat_dist.T @ P)

#print(np.linalg.matrix_power(P, 2).diagonal())
#print(np.linalg.matrix_power(P, 3).diagonal())
#print(np.linalg.matrix_power(P, 4).diagonal())

print_matrix(np.linalg.matrix_power(P, 2))
print_matrix(np.linalg.matrix_power(P, 3))
print_matrix(np.linalg.matrix_power(P, 4))
print_matrix(np.linalg.matrix_power(P, 5))

dist = np.array([
    [0, 30, 0, 0, 0, 0, 0],
    [30, 0, 40, 55, 70, 0, 0],
    [0, 40, 0, 0, 0, 0, 80],
    [0, 55, 0, 0, 0, 55, 0],
    [0, 70, 0, 0, 0, 0, 70],
    [0, 0, 0, 55, 0, 0, 20],
    [0, 0, 80, 0, 70, 20, 0]
])

res = 0
for i in range(len(P)):
    for j in range(len(P)):
        if (P[i][j] == 0):
            continue
        aux = dist[i][j] * P[i][j] * (stat_dist[i] / stat_dist[6])
        print(aux)
        res += aux

print(res)



