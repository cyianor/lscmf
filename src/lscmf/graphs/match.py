from typing import Any
from dataclasses import dataclass
from collections.abc import Hashable

from .multi import MultiEdge
from .hyper import HyperGraph


@dataclass(frozen=True)
class MatchNode:
    """Node for matching"""

    data_edge: MultiEdge[str, Hashable]
    factor: int


@dataclass(frozen=True)
class MatchEdge:
    """Edge for matching"""

    view_node: Hashable
    factor: int


class MatchGraph(HyperGraph[MatchNode, Hashable]):
    def add_relationship(
        self,
        index: int,
        match: MatchNode,
        **kwargs: Any,
    ) -> None:
        if index not in self.edges:
            self.add_edge(index)

        if match not in self.nodes:
            self.add_node(match, **kwargs)

        self[index].add(match)
