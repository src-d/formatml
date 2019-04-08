from typing import (
    Any,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)


_T = TypeVar("_T", bound=Hashable)


class Vocabulary(Generic[_T]):
    """Vocabulary utility class."""

    def __init__(
        self, unknown: Optional[_T] = None, initial_words: Tuple[_T, ...] = tuple()
    ) -> None:
        """Construct a vocabulary."""
        self.index_to_item: List[_T] = []
        self.item_to_index: Dict[_T, int] = {}
        self.unknown = unknown
        self.add_items(initial_words)
        if self.unknown is not None:
            self.add_item(self.unknown)

    def add_item(self, item: _T) -> None:
        """
        Add an item to the index.

        :param item: Item to add to the index.
        """
        if item not in self.item_to_index:
            self.item_to_index[item] = len(self.index_to_item)
            self.index_to_item.append(item)

    def add_items(self, items: Iterable[_T]) -> None:
        """
        Add items to the index.

        :param items: Items to add to the index.
        """
        for item in items:
            self.add_item(item)

    def get_index(self, item: _T) -> int:
        """
        Retrieve the index of an item.

        :param item: Item to retrieve the index of.
        :return: Index of the passed item.
        """
        if item not in self.item_to_index and self.unknown is not None:
            return self.item_to_index[self.unknown]
        return self.item_to_index[item]

    def get_indexes(self, items: Iterable[_T]) -> List[int]:
        """
        Retrieve the indexes of items.

        :param items: Items to retrieve the indexes of.
        :return: Indexes of the passed items.
        """
        return [self.get_index(item) for item in items]

    def get_item(self, index: int) -> _T:
        """
        Retrieve the item at the given index.

        :param index: Index of the item to retrieve.
        :return: Item at the given index.
        """
        return self.index_to_item[index]

    def get_items(self, indexes: Iterable[int]) -> List[_T]:
        """
        Retrieve the items at the given indexes.

        :param indexes: Indexes of the items to retrieve.
        :return: Items at the given indexes.
        """
        return [self.get_item(index) for index in indexes]

    def __contains__(self, item: Any) -> bool:
        """
        Check whether item is contained in the index.

        :param item: Item to check.
        :return: True if item is in the index, False if not.
        """
        return item in self.item_to_index

    def __len__(self) -> int:
        return len(self.item_to_index)
