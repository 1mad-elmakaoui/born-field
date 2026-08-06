"""Data-source adapters. See :mod:`born_field.adapters.base` for the contract."""

from born_field.adapters.base import Coverage, DataSourceAdapter, VectorisedSource
from born_field.adapters.pems import PeMSAdapter
from born_field.adapters.switrs import SWITRSAdapter
from born_field.adapters.synthetic import SyntheticAdapter

__all__ = [
    "Coverage",
    "DataSourceAdapter",
    "PeMSAdapter",
    "SWITRSAdapter",
    "SyntheticAdapter",
    "VectorisedSource",
]
