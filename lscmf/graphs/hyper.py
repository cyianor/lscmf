from typing import Any, Generic, TypeVar
from collections import defaultdict

from .base import PropertyDict


V = TypeVar("V")
E = TypeVar("E")


class HyperGraph(Generic[V, E]):
    """A base class for hyper graphs."""

    __slots__ = ("nodes", "edges", "graph")

    def __init__(self) -> None:
        self.graph: defaultdict[E, set[V]] = defaultdict(set)
        self.nodes: PropertyDict[V] = PropertyDict()
        self.edges: PropertyDict[E] = PropertyDict()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperGraph):
            return False

        return (
            self.graph == other.graph
            and self.nodes == other.nodes
            and self.edges == other.edges
        )

    def add_node(self, key: V, **kwargs: Any) -> None:
        self.nodes[key] = defaultdict(lambda: None, kwargs)

    def add_edge(self, key: E, **kwargs: Any) -> None:
        self.edges[key] = defaultdict(lambda: None, kwargs)

    def clear(self) -> None:
        self.graph.clear()

    def __getitem__(self, edge: E) -> set[V]:
        return self.graph[edge]

    def __setitem__(self, edge: E, value: set[V]) -> None:
        self.graph[edge] = value

    def __delitem__(self, edge: E) -> None:
        if edge in self.graph:
            del self.graph[edge]

    def incident_edges(self, key: V) -> list[E]:
        if key not in self.nodes:
            raise ValueError(f"Node {key} not in graph.")

        return [e for e, vs in self.graph.items() if key in vs]
