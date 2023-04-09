import numpy as np

dist = np.array([[0 , 0 , 0 , 0 , 0 , 30 ],
                [0 , 0 , 70, 55 , 30 , 40 ],
                [10 , 0 , 0 , 0 , 40 , 80 ],
                [10 , 0 , 0 , 0 , 55 , 55],
                [10 , 0 , 0 , 0 , 70 , 70 ],
                [0 , 0 , 0 , 0 , 55 , 20 ],
                [0 , 0 , 70 , 20 , 80 , 0 ]])

c = np.ones((dist.shape[0], dist.shape[1]))

RP, A, B, C, D, E, F = range(dist.shape[0])
COLLECT, DROP, UP, DOWN, LEFT, RIGHT = range(dist.shape[1])
GARBAGE_LOC = [B, C]
STATUS = 'Empty'
STATES = ['RP', 'A', 'B', 'C', 'D', 'E', 'F']

res = ''
cost_matrix = []
for i in range(dist.shape[0]):
    res += STATES[i] + '_{' + ''.join(STATES[i] for i in GARBAGE_LOC) + ', ' + STATUS + '} & '
    for j in range(dist.shape[1]):
        if STATUS == 'Full' and i == RP and j == DROP:
            c[i,j] = 0
        elif dist[i,j] != 0:
            if ( i in (B, C, D) and j == COLLECT and
                STATUS == 'Empty' and i not in GARBAGE_LOC):
                c[i,j] = 1
                res += str(int(c[i,j]))
            else:
                c[i,j] = len(GARBAGE_LOC) * 0.1 / 30 * dist[i,j]
                res += str(np.round(c[i,j], 3))
        else:
            res += str(int(c[i,j]))

        if j != c.shape[1]-1:
            res += ' & '
    res += ' \cr\n'

print(res)