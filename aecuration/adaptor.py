import os
import pickle

import numpy as np

from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from .base import BaseProcessor, BaseOperator

class WaveformFilterRecording(BasePreprocessor):
    """
    Removes bad events from recording extractor traces. Detected bad events are zeroed-out. Adapted to spikeinterface
    framework as a preprocessor.
    Example:
        operator = TFclassifyOperator(location of the model)
        processor = FilterProcessor(operator)
        filtered = WaveformFilterRecording(recording, processor)
    One line example:
        filtered = WaveformFilterRecording(recording, FilterProcessor(TFclassifyOperator(location of the model)))
    It is recommended to provide peak list by user defined parameters. The default peak detection is not very efficient 
    and contains a lot of noise.
    """
    name = 'waveform_filter'
    def __init__(self, recording, processor, list_peaks = None, inplace = False, verbose = False, replace = 0):
        num_seg = recording.get_num_segments()
        # some check
        assert isinstance(processor, BaseProcessor)
        if list_peaks is None:
            list_peaks = detect_peaks(recording,method='locally_exclusive')
        self.processor = processor
        sf = recording.get_sampling_frequency()
        BasePreprocessor.__init__(self, recording)
        #divide original trace into several sub trace for efficient computation
        for seg_index, parent_segment in enumerate(recording._recording_segments):
            rec_segment = WaveformFilterRecordingSegment(parent_segment, list_peaks, processor, inplace, verbose, replace)
            self.add_recording_segment(rec_segment)
        self._kwargs = dict(recording=recording.to_dict(), processor=processor,
                        list_peaks=list_peaks, inplace=inplace, verbose=verbose,
                        replace=replace)
            
class WaveformFilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, peaks, processor, inplace, verbose, replace):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.peaks = peaks
        self.processor = processor
        self.inplace = inplace
        self.verbose = verbose
        self.replace = replace
        
    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        peaks = []
        #align the time of each peak
        for i in range(len(self.peaks)):
            if self.peaks[i][0] >= start_frame and self.peaks[i][0] < end_frame:
                temp_peaks = self.peaks[i]
                temp_peaks[0] -= start_frame
                peaks.append(temp_peaks)
        peaks = np.array(peaks)
        traces = self.processor.perform(traces, peaks, self.inplace, self.verbose, self.replace)
        return traces

def WavefromFilter(*args, **kwargs):
    return WaveformFilterRecording(*args, **kwargs)