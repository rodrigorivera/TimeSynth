import torch
from torch import Tensor
import numpy as np
from typing import Callable
from .base_signal import BaseSignal

__all__ = ["PseudoPeriodic"]


class PseudoPeriodic(BaseSignal):
    """Signal generator for pseudoeriodic waves.

    The wave's amplitude and frequency have some stochasticity that
    can be set manually.

    Parameters
    ----------
    amplitude : number (default 1.0)
        Amplitude of the harmonic series
    frequency : number (default 1.0)
        Frequency of the harmonic series
    ampSD : number (default 0.1)
        Amplitude standard deviation
    freqSD : number (default 0.1)
        Frequency standard deviation
    ftype : function(default np.sin)
        Harmonic function

    """

    def __init__(
        self, amplitude:float=1.0,
            frequency:int=100,
            ampSD:float=0.1,
            freqSD:float=0.4,
            ftype:Callable[[np.array],np.array]=np.sin
    ):
        self.vectorizable = True
        self.amplitude = amplitude
        self.frequency = frequency
        self.freqSD = freqSD
        self.ampSD = ampSD
        self.ftype = ftype

    def sample_next(self, time:int, samples:Tensor, errors:Tensor)->float:
        """Sample a single time point

        Parameters
        ----------
        time : number
            Time at which a sample was required

        Returns
        -------
        float
            sampled signal for time t

        """
        freq_val = np.random.normal(loc=self.frequency, scale=self.freqSD, size=1)
        amplitude_val = np.random.normal(loc=self.amplitude, scale=self.ampSD, size=1)
        return float(amplitude_val * np.sin(freq_val * time))

    def sample_vectorized(self, times:Tensor)->Tensor:
        """Sample entire series based off of time vector

        Parameters
        ----------
        times : array-like
            Timestamps for signal generation

        Returns
        -------
        array-like
            sampled signal for time vector

        """
        n_samples = len(times)
        freq_arr = np.random.normal(
            loc=self.frequency, scale=self.freqSD, size=n_samples
        )
        amp_arr = np.random.normal(loc=self.amplitude, scale=self.ampSD, size=n_samples)
        signal = torch.tensor(np.multiply(amp_arr, self.ftype(np.multiply(freq_arr, times))))
        return signal
