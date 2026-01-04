from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities


@dataclass(frozen=True)
class ProvincialityResult:
    modularity: float | None
    n_occurrences: int
    n_unique_edges: int
    n_localities: int
    n_genera: int


def compute_locality_network_modularity(
    df: pd.DataFrame,
    *,
    locality_col: str = "locality",
    genus_col: str = "genus",
    min_localities: int = 5,
    min_genera: int = 5,
) -> ProvincialityResult:
    """
    Compute network modularity ("provinciality") for a locality co-occurrence network.

    Construction:
    - Build a bipartite graph: Locality ↔ Genus (deduplicated edges).
    - Project onto localities with weighted edges (# of shared genera).
    - Compute greedy modularity communities and modularity score (weighted).
    """
    edges_df = (
        df[[locality_col, genus_col]]
        .dropna()
        .drop_duplicates()
        .rename(columns={locality_col: "locality", genus_col: "genus"})
    )

    n_unique_edges = int(len(edges_df))
    localities = edges_df["locality"].unique()
    genera = edges_df["genus"].unique()

    if len(localities) < min_localities or len(genera) < min_genera:
        return ProvincialityResult(
            modularity=None,
            n_occurrences=int(len(df)),
            n_unique_edges=n_unique_edges,
            n_localities=int(len(localities)),
            n_genera=int(len(genera)),
        )

    G = nx.Graph()
    G.add_nodes_from(localities, bipartite=0)
    G.add_nodes_from(genera, bipartite=1)
    G.add_edges_from(edges_df.itertuples(index=False, name=None))

    locality_nodes = set(localities)

    try:
        locality_graph = nx.bipartite.weighted_projected_graph(G, locality_nodes)
        if locality_graph.number_of_edges() == 0:
            return ProvincialityResult(
                modularity=None,
                n_occurrences=int(len(df)),
                n_unique_edges=n_unique_edges,
                n_localities=int(len(localities)),
                n_genera=int(len(genera)),
            )

        communities = greedy_modularity_communities(locality_graph, weight="weight")
        modularity = nx.community.modularity(locality_graph, communities, weight="weight")
    except Exception:
        modularity = None

    return ProvincialityResult(
        modularity=float(modularity) if modularity is not None else None,
        n_occurrences=int(len(df)),
        n_unique_edges=n_unique_edges,
        n_localities=int(len(localities)),
        n_genera=int(len(genera)),
    )

