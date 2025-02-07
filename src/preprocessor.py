import torch
import torchaudio

from typing import Optional, List

class AudioPreprocessor:
    """
    A class to preprocess audio waveform by performing operations like resampling,
    truncation, and conversion to mono.

    Attributes:
        resample_rate (int): The target sample rate to which the audio should be resampled.
        to_mono (bool): Whether to convert stereo audio to mono. Default is True.
        sec_to_sample (tp.Optional[int]): Optional; The number of seconds of audio to keep after truncation.
        start_sec (tp.Optional[int]): Optional; The number of seconds to skip from the start of the audio.
    """
    def __init__(self,
                 resample_rate: int,
                 to_mono: bool = True,
                 sec_to_sample: Optional[int] = None,
                 start_sec: Optional[int] = 0,
                 chunk_duration: Optional[int] = None
                 ):
        self.resample_rate = resample_rate
        self.to_mono = to_mono
        self.sec_to_sample = sec_to_sample
        self.start_sec = start_sec
        self.chunk_duration = chunk_duration
        self._original_sample_rate = None        

    @property
    def original_sample_rate(self) -> int:
        return self._original_sample_rate

    def run(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert len(waveform.shape) == 2, "waveform should have shape of (C, S), where C = channel, S = sequence"
        self._original_sample_rate = sample_rate
        waveform = self._truncate(waveform)
        if self.to_mono:
            waveform = self._to_mono(waveform)
        waveform = self._resample(waveform)

        if self.chunk_duration:
            waveform = self._chunk_waveform(waveform, self.chunk_duration, self.resample_rate)
        else:
            waveform = waveform.unsqueeze(0)
        return waveform

    def _truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        total_num_samples = waveform.shape[1]
        starting_sample = self.original_sample_rate * self.start_sec
        assert starting_sample < total_num_samples, "starting sec is larger than the audio file's total length"
        waveform = waveform[:, starting_sample:]  # truncate the waveform to start from self.start_sec

        if self.sec_to_sample:
            num_samples_to_keep = self.original_sample_rate * self.sec_to_sample
            num_samples_to_keep = min(num_samples_to_keep, waveform.shape[1])
            waveform = waveform[:, :num_samples_to_keep]
        return waveform

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] == 2:
            mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            mono_waveform = waveform
        return mono_waveform

    def _resample(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.original_sample_rate != self.resample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=self.original_sample_rate,
                new_freq=self.resample_rate)(waveform)
        return waveform
    
    def _chunk_waveform(self, waveform: torch.Tensor, sample_rate: int, chunk_duration: int) -> List[torch.Tensor]:
        """
        Splits the audio waveform into chunks of specified duration. 
        Note that the last chunk will be discarded if it's shorter than the chunk_duration.

        Args:
            waveform (torch.Tensor): The input waveform tensor of shape (C, S).
            chunk_duration (int): The duration of each chunk in seconds.

        Returns:
            List[torch.Tensor]: A list of waveform chunks, each of shape (C, chunk_samples).
        """
        chunk_samples = sample_rate * chunk_duration
        total_samples = waveform.shape[1]

        chunks = [
            waveform[:, i:i + chunk_samples]
            for i in range(0, total_samples - chunk_samples + 1, chunk_samples)
        ]

        return torch.stack(chunks, dim=0)
