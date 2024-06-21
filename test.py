"""From Bradley, Hax and Maganti, 'Applied Mathematical Programming', figure 8.1."""

import numpy as np
import torch
import pickle
# from collections import
from pprint import pprint
from itertools import chain
# with open("dataset/pkl/circuit0.pkl",encoding="utf8") as file:
#     data = pickle.load(file)
#     print(data)
data = pickle.load(open("dataset/pkl/circuit0.pkl", "rb"))
# pprint(list(data.values()))
# flatten a list of lists
print(np.array(list(chain.from_iterable(data.values()))).reshape(-1,2))

# def Hungarian_Order(g1b, g2b):
#     from scipy.optimize import linear_sum_assignment

#     # cost matrix
#     T = np.zeros((len(g1b), len(g1b[0]), len(g1b[0])))
#     for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
#         for i, ix in enumerate(g1):
#             for j, jx in enumerate(g2):
#                 T[idx][i][j] = torch.square(
#                     ix - jx).sum()
#         row_ind, col_ind = linear_sum_assignment(T[idx])

#         g2b[idx] = g2b[idx][col_ind]

#     return g2b


# from ortools.graph.python import min_cost_flow


# def Hungarian_Order_ortool(g1b, g2b):
#     dim = len(g1b[0])
#     start_nodes = np.array([[i] * dim for i in range(dim)]).flatten()
#     end_nodes = np.array(list(range(dim)) * dim) + dim
#     capacities = np.array([1] * dim * dim)
#     # Define an array of supplies at each node.
#     supplies = [1] * dim + [-1] * dim
#     T = np.zeros((len(g1b), dim, dim))
#     for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
#         """MinCostFlow simple interface example."""
#         # Instantiate a SimpleMinCostFlow solver.
#         smcf = min_cost_flow.SimpleMinCostFlow()

#         # Define four parallel arrays: sources, destinations, capacities,
#         # and unit costs between each pair. For instance, the arc from node 0
#         # to node 1 has a capacity of 15.
#         for i, ix in enumerate(g1):
#             for j, jx in enumerate(g2):
#                 T[idx][i][j] = torch.square(ix - jx).sum()
#         unit_costs = T[idx].flatten()

#         # Add arcs, capacities and costs in bulk using numpy.
#         all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
#             start_nodes, end_nodes, capacities, unit_costs
#         )

#         # Add supply for each nodes.
#         smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

#         # Find the min cost flow.
#         status = smcf.solve()

#         if status != smcf.OPTIMAL:
#             print("There was an issue with the min cost flow input.")
#             print(f"Status: {status}")
#             exit(1)

#         solution_flows = smcf.flows(all_arcs)
#         col_ind = []
#         for arc, flow in zip(all_arcs, solution_flows):
#             if flow > 0:
#                 col_ind.append(smcf.head(arc) - dim)

#         g2b[idx] = g2b[idx][col_ind]
#     return g2b


# def Hungarian_Order_all_ortool(g1b, g2b):
#     dim = len(g1b[0])
#     start_nodes = np.repeat(np.arange(len(g1b) * dim).reshape(-1, 1), dim)
#     end_nodes = np.arange(len(g1b) * dim).reshape(-1, dim).repeat(3,
#                                                                   axis=0).flatten() + dim * len(g1b)
#     capacities = np.ones(len(g1b) * dim * dim, dtype=int)
#     T = np.zeros((len(g1b), dim, dim))
#     for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
#         for i, ix in enumerate(g1):
#             for j, jx in enumerate(g2):
#                 T[idx][i][j] = torch.square(ix - jx).sum()
#     T = T.flatten()
#     supplies = np.repeat([1, -1], len(g1b) * dim)
#     smcf = min_cost_flow.SimpleMinCostFlow()
#     unit_costs = T
#     # unit_costs = T[idx].flatten()
#     # Instantiate a SimpleMinCostFlow solver.

#     # Define four parallel arrays: sources, destinations, capacities,
#     # and unit costs between each pair. For instance, the arc from node 0
#     # to node 1 has a capacity of 15.

#     # Add arcs, capacities and costs in bulk using numpy.
#     all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
#         start_nodes, end_nodes, capacities, unit_costs
#     )
#     # # Add supply for each nodes.
#     smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)
#     # Find the min cost flow.
#     status = smcf.solve()
#     # print(start_nodes)
#     # print(end_nodes)
#     # print(capacities)
#     # print(supplies)
#     # print(unit_costs)
#     if status != smcf.OPTIMAL:
#         print("There was an issue with the min cost flow input.")
#         print(f"Status: {status}")
#         exit(1)

#     # solution_flows = smcf.flows(all_arcs)
#     # col_ind = []
#     # for arc, flow in zip(all_arcs, solution_flows):
#     #     if flow > 0:
#     #         col_ind.append(smcf.head(arc) - dim)

#     # g2b[idx] = g2b[idx][col_ind]


# size = 10000
# a = (np.random.rand(size, 3, 2) * 10).astype(int).astype(float)
# b = (np.random.rand(size, 3, 2) * 10).astype(int).astype(float)
# # a = np.array([
# #     [
# #         [1,1],
# #         [2,2],
# #         [10,10],
# #         [0,0]
# #     ]
# # ])
# # b = np.array([
# #     [
# #         [0,0],
# #         [8,8],
# #         [2,2],
# #         [1,1],
# #     ]
# # ])
# first = Hungarian_Order(a.copy(), b.copy())
# second = Hungarian_Order_all_ortool(a.copy(), b.copy())
# # print(np.square(a[0] - first).sum())
# # print(np.square(a[0] - second).sum())
# # print((first == second).all())
# # print(first)
# # print(second)
# %timeit Hungarian_Order(a.copy(), b.copy())
# %timeit Hungarian_Order_ortool(a.copy(), b.copy())
