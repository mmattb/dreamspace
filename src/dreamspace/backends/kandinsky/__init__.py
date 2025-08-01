"""Kandinsky backend package."""

from .local_backend import LocalKandinskyBackend
from .kandinsky21_server_backend import Kandinsky21ServerBackend

__all__ = ["LocalKandinskyBackend", "Kandinsky21ServerBackend"]
