"""From Bradley, Hax and Maganti, 'Applied Mathematical Programming', figure 8.1."""
import numpy as np

import torch
def Hungarian_Order(g1b, g2b):
    from scipy.optimize import linear_sum_assignment
    # cost matrix
    T=np.zeros((len(g1b),len(g1b[0]),len(g1b[0])))
    for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
        for i, ix in enumerate(g1):
            for j, jx in enumerate(g2):
                # T[idx][i][j] = torch.square(
                # ix-jx).sum()
                T[idx][i][j] = torch.sum(ix-jx).square()
        row_ind, col_ind = linear_sum_assignment(T[idx])

        g2b[idx] = g2b[idx][col_ind]

    return g2b

from ortools.graph.python import min_cost_flow
def Hungarian_Order_ortool(g1b, g2b):
    for idx, (g1, g2) in enumerate(zip(torch.as_tensor(g1b), torch.as_tensor(g2b))):
        dim = len(g1)
        """MinCostFlow simple interface example."""
        # Instantiate a SimpleMinCostFlow solver.
        smcf = min_cost_flow.SimpleMinCostFlow()

        # Define four parallel arrays: sources, destinations, capacities,
        # and unit costs between each pair. For instance, the arc from node 0
        # to node 1 has a capacity of 15.
        start_nodes = np.array([[i]*dim for i in range(dim)]).flatten()
        end_nodes = np.array(list(range(dim))*dim)+dim
        capacities = np.array([1]*dim*dim)
        T = np.zeros((dim,dim))
        for i, ix in enumerate(g1):
            for j, jx in enumerate(g2):
                # T[i][j] = torch.square(ix-jx).sum()
                T[i][j] = torch.sum(ix-jx).square()
        unit_costs = T.flatten()
        # Define an array of supplies at each node.
        supplies = [1]*dim+[-1]*dim

        # Add arcs, capacities and costs in bulk using numpy.
        all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
            start_nodes, end_nodes, capacities, unit_costs
        )

        # Add supply for each nodes.
        smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

        # Find the min cost flow.
        status = smcf.solve()

        if status != smcf.OPTIMAL:
            print("There was an issue with the min cost flow input.")
            print(f"Status: {status}")
            exit(1)

        solution_flows = smcf.flows(all_arcs)
        costs = solution_flows * unit_costs
        col_ind = []
        for arc, flow, cost in zip(all_arcs, solution_flows, costs):
            if flow>0:
                col_ind.append(smcf.head(arc)-dim)

        g2b[idx] = g2b[idx][col_ind]
    return g2b

a = (np.random.rand(15,6,1)*10).astype(int).astype(float)
b = (np.random.rand(15,6,1)*10).astype(int).astype(float)
# a = np.array([
#     [
#         [1,1],
#         [2,2],
#         [10,10],
#         [0,0]
#     ]
# ])
# b = np.array([
#     [
#         [0,0],
#         [8,8],
#         [2,2],
#         [1,1],
#     ]
# ])
one = Hungarian_Order(a.copy(),b.copy()).astype(float)
print("----------------------")
print()
two = Hungarian_Order_ortool(a.copy(),b.copy()).astype(float)
a = torch.tensor(a)
b = torch.tensor(b)
one = torch.tensor(one)
two = torch.tensor(two)

print(torch.nn.MSELoss()(a,b))
print(torch.nn.MSELoss()(a,one))
print(torch.nn.MSELoss()(a,two))