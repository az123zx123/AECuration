import os
import pickle

import numpy as np

from .base import BaseOperator

class classifyOperator(BaseOperator):
    """
    classify operator implemented by numpy
    """
    def __init__(self, model = None):
        self.model = model

    def spikeClassification(self, spike:np.array) -> np.array:
        """
        Calculate the MSLE of given spikes.

        Parameters
        ----------
        spike : np.array
            spike data. (N, M) array

        Returns
        -------
        np.array
            MSLE value for each spike. (N, ) array

        """
        x = spike.astype(np.float32)
        for i in range(len(self.model)):
            weights, bias =self.model[i]
            x = np.dot(x, weights) + bias
        return self.MSLE(x, spike.astype(np.float32))
    
    def setModel(self, model):
        self.model = model
        
    def getSize(self):
        if self.model is not None:
            weights, bias =self.model[0]
            return weights.shape[0]
        else:
            print("Find no model!")
            raise AttributeError

class TFclassifyOperator(BaseOperator):
    """
    classify operator implemented by tensorflow
    """
    @classmethod
    def isInstalled(cls):
        """
        Check if tensorflow is installed

        Returns
        -------
        HAVE_TF : bool
            Have Installed tensorflow or not.

        """
        try:
            from tensorflow.keras.models import Model, load_model
            from tensorflow.keras import losses
            HAVE_TF = True
        except ImportError:
            HAVE_TF = False
        return HAVE_TF

    def __init__(self, model:str):
        self.HAVE_TF = TFclassifyOperator.isInstalled()
        if self.HAVE_TF:
            from tensorflow.keras.models import Model, load_model
            from tensorflow.keras import losses
            self.model = load_model(model)
            self.loss = losses.MSLE
        else:
            self.model = None
            self.loss = None
    
    def spikeClassification(self, spike:np.array) -> np.array:
        """
        Calculate the MSLE of given spikes by tensorflow built-in functions        

        Parameters
        ----------
        spike : np.array
            spike data. (N, M) array.

        Returns
        -------
        np.array
            MSLE value for each spike. (N, ) array

        """
        if self.HAVE_TF:
            predict_spike = self.model.predict(spike)
            return self.loss(np.abs(predict_spike),np.abs(spike)).numpy()
        else:
            return None
        
    def setModel(self, model):
        if self.HAVE_TF:
            from tensorflow.keras.models import Model, load_model
            from tensorflow.keras import losses
            self.model = load_model(model)
            self.loss = losses.MSLE
            
    def getSize(self):
        if self.model is not None:
            weights, bias =self.model.layers[1].weights
            return weights.shape[0]
        else:
            print("Find no model!")
            raise AttributeError