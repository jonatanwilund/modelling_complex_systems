"""
* smallWorldWattsStrogatz.py
*
* Copyright (c) 2026, Jordi-Lluis Figueras
*
* OpenAI Codex / ChatGPT 5.5 has been used in the editing of this file.
*
"""

"""Explore clustering and path lengths in Watts-Strogatz networks.

Usage:
  python3 smallWorldWattsStrogatz.py
  python3 smallWorldWattsStrogatz.py --nodes 100 --neighbors 4 --trials 30
  python3 smallWorldWattsStrogatz.py --probabilities 0 0.01 0.05 0.1 0.5 1

The script prints average statistics over several random trials. The small-world
effect is visible when a small rewiring probability greatly reduces path length
while clustering remains relatively high.
"""

import argparse
import statistics
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError as error:
    raise SystemExit(
        "This script requires NetworkX. On Debian/Ubuntu, use a virtual environment:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  python3 -m pip install networkx"
    ) from error


defaultNodes = 200
defaultNeighbors = 6
defaultTrials = 1000
defaultSeed = 2026
defaultProbabilities = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Explore the Watts-Strogatz small-world model"
    )
    parser.add_argument(
        "--nodes", type=int, default=defaultNodes, help="number of nodes"
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=defaultNeighbors,
        help="number of ring neighbors per node",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=defaultTrials,
        help="number of random trials per probability",
    )
    parser.add_argument(
        "--seed", type=int, default=defaultSeed, help="base random seed"
    )
    parser.add_argument(
        "--probabilities",
        nargs="+",
        type=float,
        default=defaultProbabilities,
        help="rewiring probabilities to test",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="number of decimals in printed statistics",
    )
    args = parser.parse_args()

    if args.nodes <= 2:
        parser.error("--nodes must be larger than 2")

    if args.neighbors <= 0:
        parser.error("--neighbors must be positive")

    if args.neighbors >= args.nodes:
        parser.error("--neighbors must be smaller than --nodes")

    if args.neighbors % 2 != 0:
        parser.error(
            "--neighbors must be even for the Watts-Strogatz ring construction"
        )

    if args.trials <= 0:
        parser.error("--trials must be positive")

    if args.decimals < 0:
        parser.error("--decimals must be nonnegative")

    for probability in args.probabilities:
        if probability < 0 or probability > 1:
            parser.error("each rewiring probability must satisfy 0 <= p <= 1")

    return args


def largestConnectedSubgraph(graph):
    if nx.is_connected(graph):
        return graph, 1.0

    largestComponent = max(nx.connected_components(graph), key=len)
    componentFraction = len(largestComponent) / graph.number_of_nodes()
    return graph.subgraph(largestComponent).copy(), componentFraction


def computeTrialStatistics(nNodes, nNeighbors, probability, seed):
    graph = nx.watts_strogatz_graph(nNodes, nNeighbors, probability, seed=seed)

    connectedGraph, componentFraction = largestConnectedSubgraph(graph)
    return {
        "number components": nx.number_connected_components(graph),
        "largest component": len(max(nx.connected_components(graph), key=len)),
        "diameter": nx.diameter(connectedGraph),
        "pathLength": nx.average_shortest_path_length(connectedGraph),
        "clustering": nx.average_clustering(graph),
        # "componentFraction": componentFraction,
    }


def meanStatistic(statisticsList, key):
    return statistics.mean(item[key] for item in statisticsList)


def printHeader(args):
    print("Watts-Strogatz small-world experiment")
    print(f"  nodes: {args.nodes}")
    print(f"  neighbors per node in initial ring: {args.neighbors}")
    print(f"  trials per probability: {args.trials}")
    print()
    print(
        "p        number components   largest component   diameter   path length   clustering"
    )
    print(
        "------------------------------------------------------------------------------------"
    )


def printStatisticsRow(probability, statisticsList, decimals):
    number_components = meanStatistic(statisticsList, "number components")
    largest_component = meanStatistic(statisticsList, "largest component")
    diameter = meanStatistic(statisticsList, "diameter")
    pathLength = meanStatistic(statisticsList, "pathLength")
    clustering = meanStatistic(statisticsList, "clustering")
    # componentFraction = meanStatistic(statisticsList, "componentFraction")
    print(
        f"{probability:<8.3g} "
        f"{number_components:<19.{decimals}f} "
        f"{largest_component:<19.{decimals}f} "
        f"{diameter:<10.{decimals}f} "
        f"{pathLength:<13.{decimals}f} "
        f"{clustering:<12.{decimals}f} "
        # f"{componentFraction:>18.{decimals}f}"
    )


def printInterpretation():
    print("\nInterpretation:")
    print(
        "  For p = 0 the graph is highly clustered but distances are relatively long."
    )
    print(
        "  For small positive p, shortcuts usually reduce distances faster than they destroy clustering."
    )
    print(
        "  For p close to 1, the graph behaves more like a random graph: short paths, lower clustering."
    )


def plot(path_lengths, clusterings):
    path_lengths_norm = [p / min(path_lengths) for p in path_lengths]
    clusterings_norm = [c / min(clusterings) for c in clusterings]

    plt.plot(defaultProbabilities, path_lengths_norm)
    # plt.xscale('log')
    plt.xlabel("Probabilities")
    plt.ylabel("Normalized path lengths")
    plt.show()

    plt.plot(defaultProbabilities, clusterings_norm)
    # plt.xscale('log')
    plt.xlabel("Probabilities")
    plt.ylabel("Normalized clustering")
    plt.show()


def main():
    args = parseArguments()
    printHeader(args)

    path_lengths = []
    clusterings = []

    for probabilityIndex, probability in enumerate(args.probabilities):
        statisticsList = []
        for trial in range(args.trials):
            seed = args.seed + 1000 * probabilityIndex + trial
            trialStatistics = computeTrialStatistics(
                args.nodes, args.neighbors, probability, seed
            )
            statisticsList.append(trialStatistics)

        printStatisticsRow(probability, statisticsList, args.decimals)

        path_lengths.append(meanStatistic(statisticsList, "pathLength"))
        clusterings.append(meanStatistic(statisticsList, "clustering"))

    printInterpretation()

    plot(path_lengths, clusterings)


if __name__ == "__main__":
    main()
