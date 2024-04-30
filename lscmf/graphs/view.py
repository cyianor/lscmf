from numpy import float64, ndim, asarray
from numpy.typing import NDArray
from typing import Iterable
from collections.abc import Hashable

from .multi import MultiEdge, UndirectedMultiGraph

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class ViewGraph(UndirectedMultiGraph[str, Hashable]):
    """A graph representing the relationships among views and their data.

    Nodes represent different views and edges represent relationships between
    views. Data matrices are therefore associated with edges. Multiple
    relationships between the same pair of views can be represented by
    multi-edges.
    """

    def add_data(
        self, name: str, x: NDArray[float64], views: tuple[Hashable, Hashable]
    ) -> None:
        """Add data for a view-relationship to the graph.

        This function adds a new data matrix for a view-relationship to
        the graph.

        The function also performs consistency checks on the size of
        views, so they are compatible among different nodes.

        The nodes store the views and their associated size as property ``p``.
        The edges store the data as property ``x``, .

        In addition, the ratio of rows to columns of the data matrix
        is computed and saved in the node property ``beta``.
        If ``beta > 1.0`` then ``beta = 1.0 / beta`` and the edge attribute
        ``transposed`` will be ``True``.

        Parameters
        ----------
        name : str
            A unique name for the relationship.
        x : ndarray[float64]
            A matrix containing the data for the relationship.
        views : (hashable, hashable)
            The indices describing the views of the relationship.

        Raises
        ------
        ValueError
            If invalid data is passed. This can be views describing a
            relationship of a view with itself, an already existing view
            relationship, or if the matrix dimensions are not consistent
            with previously added dimensions.
        TypeError
            If ``x`` is not a matrix.
        """
        if ndim(x) != 2:
            raise TypeError("x needs to be a matrix")

        if views[0] == views[1]:
            raise ValueError(
                f"Adding a view-relationship relating view {views[0]}"
                " to itself is not possible."
            )

        # Avoid adding the same relationship twice.
        if MultiEdge(name, views) in self.edges:
            raise ValueError(
                f"View-relationship {views} with name {name} exists already."
            )

        beta = x.shape[0] / x.shape[1]
        transposed = False
        # If number of rows > number of columns
        if beta > 1.0:
            beta = 1.0 / beta
            transposed = True

        for i, v in enumerate(views):
            if v not in self.nodes:
                self.add_node(v, p=x.shape[i])
            else:
                if self.nodes[v]["p"] != x.shape[i]:
                    raise ValueError(
                        f"View {v} previously had size {self.nodes[v]['p']}."
                        f"Got {x.shape[i]} instead."
                    )

        self.add_edge(
            name,
            views,
            x=asarray(x, dtype=float64),
            beta=beta,
            transposed=transposed,
            views=views,  # Correspondence of matrix dimensions to views
        )

    def add_data_from(
        self,
        names: Iterable[str],
        xs: Iterable[NDArray[float64]],
        viewrels: Iterable[tuple[Hashable, Hashable]],
    ) -> None:
        """Add multiple data sources at once.

        This is a convenience wrapper to ``add_data()``. ``xs`` and
        ``viewrels`` need to be provided in the same order as ``names``.

        Parameters
        ----------
        names : iterable of ``n`` str
            Names for the data sources
        xs : iterable of ``n`` ndarray[float64]
            Data matrices
        viewrels : iterable of ``n`` (hashable, hashable)
            View-relationships
        """
        for name, x, views in zip(names, xs, viewrels):
            self.add_data(name=name, x=x, views=views)

    def focus(self, view: Hashable) -> Self:
        """Return a sub-graph with focus on a single view.

        Parameters
        ----------
        view : hashable
            The view to be focused on.

        Returns
        -------
        G : ViewGraph
            A subgraph containing only data related to the requested view.
        """
        # Find all views that are related to `view`
        nodes = set(
            e.n[1] if e.n[0] == view else e.n[0]
            for e in self.incident_edges(view)
        )
        # Add `view` itself
        nodes.add(view)

        G = self.subgraph(nodes)

        return G
