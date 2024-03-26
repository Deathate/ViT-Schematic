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
                T[idx][i][j] = torch.square(
                ix-jx).sum()
        row_ind, col_ind = linear_sum_assignment(T[idx])
        g2b[idx] = g2b[idx][col_ind]

    return g2b

from ortools.graph.python import min_cost_flow
def Hungarian_Order_ortool(g1b, g2b):
    """MinCostFlow simple interface example."""
    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()

    # Define four parallel arrays: sources, destinations, capacities,
    # and unit costs between each pair. For instance, the arc from node 0
    # to node 1 has a capacity of 15.
    start_nodes = np.array([0, 0, 1, 1])
    end_nodes = np.array([2, 2, 3, 3])
    capacities = np.array([1, 1, 1, 1])
    unit_costs = np.array([2,3,4,5])

    # Define an array of supplies at each node.
    supplies = [1, 1,-1,-1]

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
    for arc, flow, cost in zip(all_arcs, solution_flows, costs):
        if flow>0:
            print(smcf.tail(arc), smcf.head(arc))


if __name__ == "__main__":
    main()