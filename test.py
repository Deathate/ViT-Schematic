import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from utility import *

# Example: Set A and Set B each have 3 lines (defined by start and end points)
set_a = [((0, 0), (1, 1)), ((2, 3), (4, 5)), ((1, 2), (3, 4))]
set_b = [((1, 1), (2, 2)), ((0, 0), (1, 2)), ((2, 3), (4, 6))]


# Define the cost function (Euclidean distance between line endpoints)
def calculate_cost(line_a, line_b):
    cost = snorm2(line_a[0], line_b[0]) + snorm2(line_a[1], line_b[1])
    return cost


# Create the cost matrix
num_lines = len(set_a)
cost_matrix = np.zeros((num_lines, num_lines))

for i in range(num_lines):
    for j in range(num_lines):
        cost_matrix[i, j] = calculate_cost(set_a[i], set_b[j])

# Solve the assignment problem
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Output the optimal assignment
print("Optimal Assignment:")
for i, j in zip(row_ind, col_ind):
    print(f"Line {i} in Set A is assigned to Line {j} in Set B with cost {cost_matrix[i, j]}")
print(benchmark(lambda: norm2((1, 2), (3, 4))))
