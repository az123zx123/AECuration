import os
import pickle
from packaging.version import parse
import warnings
from abc import ABC, abstractmethod

import numpy as np

class BaseOperator(ABC):
    """
    Interface for all the operator, perform the spike classification function.
    """
    @abstractmethod
    def __init__(self, *args,**kwargs):
        pass
    
    @abstractmethod
    def spikeClassification(self, *args,**kwargs):
        pass
    
    @abstractmethod
    def setModel(self, *args,**kwargs):
        pass
    
    @abstractmethod
    def getSize(self):
        pass
    
    @classmethod
    def MSLE(cls, y:np.array, y0:np.array) -> np.array:
        """
        Computes the mean squared logarithmic error. Since spike can be negative, take the absolute value

        Parameters
        ----------
        y : Numpy Array
        y0 : Numpy Array

        Returns
        -------
        np.array
            MSLE value.

        """
        return np.mean(np.square(np.log1p(np.abs(y)) - np.log1p(np.abs(y0))), axis=-1)
    

class BaseProcessor(ABC):
    """
    Interface for all the processor, perform spesific part in the pipeline.
    """
    
    @abstractmethod
    def __init__(self, *args,**kwargs):
        pass
    
    @abstractmethod
    def perform(self, *args,**kwargs):
        pass
    
    @abstractmethod
    def setModel(self, *args,**kwargs):
        pass
