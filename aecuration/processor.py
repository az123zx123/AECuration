import os
import pickle
from packaging.version import parse
import warnings
from tqdm import tqdm

import numpy as np

from .base import BaseProcessor
class FilterProcessor(BaseProcessor):
    """
    batch waveform filter: input is the trace and detected peaks. 
    Removing peaks by setting to a constant if MSLE value is larger than a threhsold.
    """
    def __init__(self, operator = None, threshold = 1.5):
        self.operator = operator
        self.threshold = threshold
        
    def perform(self, trace:np.array, peaks:np.array, inplace = False, verbose = False, replace = 0) -> np.array:
        """
        Perform batch waveform filter. Extract waveforms from the raw trace and combine into one matrix.
        Operator classification based on the matrix.
        Replace trace with a constant if the score is higher than a threshold
        Parameters
        ----------
        trace : np.array
            raw trace.
        peaks : np.array
            detected peaks. Each raw is a detected peak. 
            First value is the preak location, second value is the channel.
        inplace : bool, optional
            Change the input trace. The default is False.
        verbose : bool, optional
            show details. The default is False.
        replace : bool, optional
            The replace constant. The default is 0.

        Returns
        -------
        filter_trace : np.array
            filtered trace.

        """
        #get size of each time interval
        try:
            size = self.operator.getSize()
        except AttributeError:
            print("Wrong setting. Check operator")
            return
        spikes = []
        index = []
        if not verbose:
            if not inplace:
                filter_trace = np.copy(trace) #copy the input trace into a new array
            else:
                filter_trace = trace #direct change the input trace
            #Extract potential spike events from trace and combine into one matrix
            for i in range(peaks.shape[0]):
                loc = peaks[i][0]
                channel = peaks[i][1]
                if loc > size/2 and loc + size/2 < trace.shape[0]:
                    spikes.append(trace[loc-int(size/2):loc+int(size/2),channel])
                    index.append(loc)
            spikes = np.array(spikes)
            spikes = spikes.reshape((len(index),size))
            score = self.operator.spikeClassification(spikes)
            check = score>self.threshold
            #replace detected noise events with the replace value
            for i in range(len(index)):
                if not check[i]:
                    filter_trace[index[i]-int(size/2):index[i]+int(size/2),:] = replace
        else: #print details
            if not inplace:
                print("copy trace")
                filter_trace = np.copy(trace) #copy the input trace into a new array
            else:
                filter_trace = trace #direct change the input trace
            #Extract potential spike events from trace and combine into one matrix
            print("extract waveform")
            for i in tqdm(range(peaks.shape[0])):
                loc = peaks[i][0]
                channel = peaks[i][1]
                if loc > size/2 and loc + size/2 < trace.shape[0]:
                    spikes.append(trace[loc-int(size/2):loc+int(size/2),channel])
                    index.append(loc)
            spikes = np.array(spikes)
            score = self.operator.spikeClassification(spikes)
            check = score>self.threshold
            #replace detected noise events with the replace value
            print("filtering trace")
            for i in tqdm(range(len(index))):
                if not check[i]:
                    filter_trace[index[i]-int(size/2):index[i]+int(size/2),:] = replace
        return filter_trace
            
    def setModel(self, operator = None):
        self.operator = operator
    
    def setThreshold(self, threshold = 1.5):
        self.threshold = threshold

class CurationProcessor(BaseProcessor):
    """
    Batch curation processor: input is the waveforms of each unit, output is the class of each unit
    strategy:
        both: evalute both template and mean. Pass both is good, fail one is hybrid, fail both is bad
        template: evalute only template score
        mean: evalute only mean score
    """
    
    def __init__(self, operator = None, template_threshold = 1.5, mean_threshold = 1.5, strategy="both"):
        self.operator = operator
        self.template_threshold = template_threshold
        self.mean_threshold = mean_threshold
        self.strategy = strategy
        
    def perform(self, waveforms:list, verbose = False) -> np.array:
        """
        Perform batch curation. Input is a list of waveforms. Each compotent is a matrix contains waveform from each unit.
        The length of waveforms is the range of unit id.

        Parameters
        ----------
        waveforms : list
            List of waveform. Each item is a numpy array (N, M)
        verbose : bool, optional
            show details. The default is False.

        Returns
        -------
        classes : numpy array
            1: good unit
            0: bad unit
            -1: hybrid unit.

        """
        classes = np.zeros((len(waveforms,)))
        if not verbose:
            for unit_id in range(len(waveforms)):
                waveform = waveforms[unit_id]
                template_score = self.operator.spikeClassification(np.mean(waveform,axis=0))
                mean_score = np.mean(self.operator.spikeClassification(waveform))
                if self.strategy == "both":
                    if template_score < self.template_threshold and mean_score < self.mean_threshold:
                        classes[unit_id] = 1
                    elif template_score < self.template_threshold or mean_score < self.mean_threshold:
                        classes[unit_id] = -1
                elif self.strategy == "template":
                    if template_score < self.template_threshold:
                        classes[unit_id] = 1
                elif self.strategy == "mean":
                    if mean_score < self.mean_threshold:
                        classes[unit_id] = 1
                else:
                    if template_score < self.template_threshold and mean_score < self.mean_threshold:
                        classes[unit_id] = 1
                    elif template_score < self.template_threshold or mean_score < self.mean_threshold:
                        classes[unit_id] = -1                    
        else:
            print("performing curation")
            for unit_id in tqdm(range(len(waveforms))):
                waveform = waveforms[unit_id]
                template_score = self.operator.spikeClassification(np.mean(waveform,axis=0))
                mean_score = np.mean(self.operator.spikeClassification(waveform))
                if self.strategy == "both":
                    if template_score < self.template_threshold and mean_score < self.mean_threshold:
                        classes[unit_id] = 1
                    elif template_score < self.template_threshold or mean_score < self.mean_threshold:
                        classes[unit_id] = -1
                elif self.strategy == "template":
                    if template_score < self.template_threshold:
                        classes[unit_id] = 1
                elif self.strategy == "mean":
                    if mean_score < self.mean_threshold:
                        classes[unit_id] = 1
                else:
                    if template_score < self.template_threshold and mean_score < self.mean_threshold:
                        classes[unit_id] = 1
                    elif template_score < self.template_threshold or mean_score < self.mean_threshold:
                        classes[unit_id] = -1          
        return classes, template_score, mean_score
            
    def setModel(self, operator = None):
        self.operator = operator
    
    def setTemplateThreshold(self, threshold=1.5):
        self.template_threshold = threshold
    
    def setMeanThreshold(self, threshold=1.5):
        self.mean_threshold = threshold
    
    def setThreshold(self, template_threshold=1.5, mean_threshold=1.5):
        self.setTemplateThreshold(template_threshold)
        self.setMeanThreshold(mean_threshold)

class ImproveProcessor(BaseProcessor):
    """
    Batch improvement. 
    """
    def __init__(self, operator = None, threshold = 1.5):
        self.operator = operator
        self.threshold = threshold
        
    def perform(self, waveforms:list, spike_train:list, inplace = False, verbose = False) -> np.array:
        spike_list = []
        if not verbose:
            for unit_id in range(len(waveforms)):
                waveform = waveforms[unit_id]
                score = self.operator.spikeClassification(waveform)
                index = score < self.threshold
                if inplace:
                    spike_train[unit_id] = spike_train[unit_id][index]
                    x = spike_train[unit_id]
                else:
                    x = spike_train[unit_id][index]
                spike_list.append(x)
        else:
            print("imrpove spike train")
            for unit_id in tqdm(range(len(waveforms))):
                waveform = waveforms[unit_id]
                score = self.operator.spikeClassification(waveform)
                index = score < self.threshold
                if inplace:
                    spike_train[unit_id] = spike_train[unit_id][index]
                    x = spike_train[unit_id]
                else:
                    x = spike_train[unit_id][index]
                spike_list.append(x)      
        return spike_list
        
    def setModel(self, operator = None):
        self.operator = operator
    
    def setThreshold(self, threshold = 1.5):
        self.threshold = threshold    