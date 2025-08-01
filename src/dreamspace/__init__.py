"""Dreamspace Co-Pilot: AI + BCI Co-Piloted Visual Dreamspace

A hybrid Brain-Computer Interface (BCI) and Artificial Intelligence (AI) system 
that enables users to co-pilot and explore imaginative visual experiences.
"""

from .core.image_gen import ImgGen
from .core.base import ImgGenBackend
from .config.settings import Config

__version__ = "0.1.0"
__all__ = ["ImgGen", "ImgGenBackend", "Config"]
