from typing import (
    Generic,
    TypeVar,
    Any,
    Callable,
    Generator,
)
from collections import defaultdict

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


Properties = defaultdict[str, Any]

T = TypeVar("T")


class PropertyDict(Generic[T], dict[T, Properties]):
    """A dictionary allowing easy access to properties."""

    __slots__ = ()

    def items_filtered(
        self, *properties: str
    ) -> Generator[tuple[T, tuple[Any, ...]], None, None]:
        """Iterate over (key, value)-pairs and a subset of properties.

        Parameters
        ----------
        *properties : str
            The names of the properties to be filtered by.

        Returns
        -------
        it : generator[(k, (p1, p2, ...))]
            A generator of tuples consisting of the key ``k`` and a tuple
            containing the requested properties ``p1``, ``p2``, and so on.
        """
        for n in self:
            yield n, tuple(self[n][p] for p in properties)

    def copy(self) -> Self:
        """A shallow copy of the PropertyDict."""
        return type(self)(self)

    def update_properties(
        self,
        f: Callable[..., Any],
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
    ) -> None:
        """Apply a function to one or more properties across all entries.

        Notes
        -----
        Does nothing if a property does not exist for an entry, i.e. if
        the property is ``None``. This makes it possible to apply updates
        when properties only exist on some entries.

        Parameters
        ----------
        f : callable
            Callable to be applied to ``inputs``
        inputs : tuple[str, ...]
            Property names used as inputs into ``f``. Note that these
            are passed to ``f`` in the given order.
        outputs : tuple[str, ...]
            Property names given to outputs of ``f`` and which are used
            to update the PropertyDict. Note that these are used in the
            given order.
        """
        for n in self:
            if all(self[n][p] is not None for p in inputs):
                out = f(*[self[n][p] for p in inputs])
                # If there is only one return value, package it as a tuple
                if len(outputs) == 1:
                    out = (out,)
                self[n].update(zip(outputs, out))
