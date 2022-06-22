import itertools
import textwrap
from abc import ABCMeta, abstractproperty
from typing import Generic, ItemsView, Iterator, Optional, TypeVar, final

import _collections_abc

def center_text(text: str, prefix_blank_line: bool = False, append_blank_line: bool = False,
                framing_width: int = 0, frame_before: bool = True, frame_after: bool = True, framing_char: str = '=',
                vbar_left: str = '', vbar_right: str = '', centering_width: int = 120, terminal_width: int = 160) -> str:
    "Function for generating centered text for printing to the console."
    centered_text: str = ""
    free_space: int = framing_width - (len(vbar_left) + len(vbar_right))
    line_iter = itertools.chain(*[textwrap.wrap(f"{vbar_left + (' ' * ((free_space - len(part)) // 2)):>s}{part}{(' ' * ((free_space - len(part)) // 2)) + vbar_right:<s}",
                                                width=terminal_width, replace_whitespace=False, drop_whitespace=False) for part in text.split("\n")])
    if prefix_blank_line: centered_text += "\n"
    if framing_width != 0 and frame_before: centered_text += f"{(framing_char * framing_width).center(centering_width)}\n"
    for line in line_iter: centered_text += f"{(line.center(centering_width))}\n"
    if framing_width != 0 and frame_after: centered_text += f"{(framing_char * framing_width).center(centering_width)}"
    if append_blank_line: centered_text += "\n"
    return centered_text

class AbstractionHierarchy(metaclass=ABCMeta):
    """
    Interface class for defining classes representing or containing abstraction hierarchies.
    Implementation requires overriding the single abstract property `top_level : int`.
    
    Properties
    ----------
    `top_level : int` - The top level of abstraction hierarchy.
    This should be a positive non-zero integer (a natural number).
    
    Example Usage
    -------------
    
    A minimum example for creating an abstraction hierarchy class is as follows.
    
    ```
    class SimpleHierarchy(AbstractionHierarchy):
        def __init__(self, top_level: int):
            if top_level < 1:
                raise valueError("Top level must be a non-zero positive integer")
            self.__top_level = top_level
        
        @property
        def top_level(self) -> int:
            return self.__top_level
    ```
    
    The interface methods can then be used to obtain and iterate over the abstraction hierarchy's level range.
    
    >>> sh = SimpleHierarchy(top_level=5)
    >>> sh.level_range
    range(1, 6)
    >>> levels: list[int] = []
    >>> for level in sh.level_range: levels.append(level)
    >>> levels
    [1, 2, 3, 4, 5]
    
    The level range can be easily constrained to a explicit top and bottom level which are normalised against the hierarchy.
    
    >>> sh.constrained_level_range(2, 4)
    range(2, 5)
    >>> sh.constrained_level_range(-1, 4)
    range(1, 5)
    >>> sh.constrained_level_range(3, 10)
    range(3, 6)
    
    To simply check of the hierarchy contains are level a convenience method is provided.
    
    >>> sh.in_range(3)
    True
    >>> sh.in_range(10)
    False
    
    Notes
    -----
    This class declares no instance variables and an empty `__slots__`.
    """
    __slots__ = ()
    
    @property
    def bottom_level(self) -> int:
        "An integer defining the bottom level of the abstraction hierarchy, by default 1 if not overrided."
        return 1
    
    @abstractproperty
    def top_level(self) -> int:
        "An integer defining the top level of the abstraction hierarchy."
        raise NotImplementedError("Abstraction hierarchies must have a top-level.")
    
    @final
    @property
    def level_range(self) -> range:
        "A contiguous range over all levels in the abstraction hierarchy, the range is `[bottom_level-top_level]`."
        return range(self.bottom_level, self.top_level + 1)
    
    @final
    def constrained_level_range(self, bottom_level: Optional[int] = None, top_level: Optional[int] = None) -> range:
        """
        A contiguous constrained range over the levels of the abstraction hierarchy.
        The returned range is `[max(1, bottom_level)-min(self.top_level, top_level)]`,
        and is always valid, such that all levels in the range and in the hierarchy.
        
        Parameters
        ----------
        `bottom_level : {int | None}` - The bottom-level constraint.
        
        `top_level : {int | None}` - The top-level constraint.
        If not given or None, defaults to the hierarchies' top-level.
        
        Returns
        -------
        `range` - The constrained contiguous level range.
        
        Raises
        ------
        `TypeError` - Iff either the top or bottom level constraint is not None and not an integer.
        """
        ## Validate inputs
        if bottom_level is not None and not isinstance(bottom_level, int):
            raise TypeError(f"Bottom level must be an integer or None. Got; {bottom_level=} of {type(bottom_level)=}.")
        if top_level is not None and not isinstance(top_level, int):
            raise TypeError(f"Top level must be an integer or None. Got; {top_level=} of {type(top_level)=}.")
        
        ## Extract optional inputs
        _bottom_level: int = bottom_level if bottom_level is not None else self.bottom_level
        _top_level: int = top_level if top_level is not None else self.top_level
        
        ## Normalise them
        _bottom_level = min(max(self.bottom_level, _bottom_level), self.top_level + 1)
        _top_level = max(min(_top_level + 1, self.top_level + 1), _bottom_level)
        
        return range(_bottom_level, _top_level)
    
    @final
    def in_range(self, level: int) -> bool:
        """
        Check whether an abstraction level is in the hierarchy.
        
        Parameters
        ----------
        `level : int` - A integer defining the abstraction level to check.
        
        Returns
        -------
        `bool` - A Boolean, True if the level is in the range [1-top_level], False otherwise.
        """
        return level in self.level_range

KT = TypeVar("KT")
VT = TypeVar("VT")

class ReversableDict(_collections_abc.MutableMapping, Generic[KT, VT]):
    """
    A mutable reversable dictionary type.
    Contains all usual methods of standard dictionaries as well as the ability to find all keys that map to a specific value.
    
    Example Usage
    -------------
    
    >>> from core.Helpers import ReversableDict
    >>> dict_: ReversableDict[str, int] = ReversableDict({"A" : 2, "B" : 3, "C" : 2})
    
    Check what 'A' maps to (as in a standard dictionary).
    >>> dict_["A"]
    2
    
    Check what keys map to 2 (a reverse operation to a standard dictionary).
    >>> dict_(2)
    ["A", "C"]
    
    Objects are immutable, and updates are handled correctly on both the standard and reverse mappings.
    >>> del dict_["A"]
    >>> dict_(2)
    ["C"]
    """
    __slots__ = ("__dict", "__reversed_dict")
    
    def __init__(self, dict_: dict[KT, VT] = {}) -> None:
        self.__dict: dict[KT, VT] = {}
        self.__reversed_dict: dict[VT, list[KT]] = {}
        for key, value in dict_.items():
            self[key] = value
    
    def __repr__(self) -> str:
        return repr(self.__dict)
    
    def __getitem__(self, key: KT) -> VT:
        return self.__dict[key]
    
    def __call__(self, value: VT) -> list[KT]:
        return self.__reversed_dict[value].copy()
    
    def __setitem__(self, key: KT, value: VT) -> None:
        if (old_value := self.__dict.get(key)) is not None:
            self.__del_reversed_item(key, old_value)
        self.__dict[key] = value
        self.__reversed_dict.setdefault(value, []).append(key)
    
    def __delitem__(self, key: KT) -> None:
        value: VT = self.__dict[key]
        del self.__dict[key]
        self.__del_reversed_item(key, value)
    
    def __del_reversed_item(self, key: KT, value: VT) -> None:
        self.__reversed_dict[value].remove(key)
        if len(self.__reversed_dict[value]) == 0:
            del self.__reversed_dict[value]
    
    def __iter__(self) -> Iterator[KT]:
        yield from self.__dict
    
    def __len__(self) -> int:
        return len(self.__dict)
    
    def reversed_items(self) -> ItemsView[VT, list[KT]]:
        """
        Get a reversed dictionary items view:
            - Whose keys are the values of the standard dictionary,
            - And whose values are lists of keys from the standard dictionary which map to the respective values.
        """
        return self.__reversed_dict.items()
    
    def reversed_get(self, value: VT, default: Optional[list[KT]] = None) -> Optional[list[KT]]:
        "Get a list of dictionary keys that map to a given value, default is returned of the value is not in the dictionary."
        if value not in self.__reversed_dict:
            return default
        return self(value)
