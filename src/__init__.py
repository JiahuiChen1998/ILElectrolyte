"""
ILElectrolyte: Molecular Property Prediction for Ionic Liquids

A package for predicting electrochemical properties of ionic liquids
using machine learning models.
"""

__version__ = "0.1.0"
__author__ = "ILElectrolyte Team"

from . import model
from . import data_loader
from . import utils

__all__ = ["model", "data_loader", "utils"]
