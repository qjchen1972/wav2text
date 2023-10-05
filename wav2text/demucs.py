# -*- coding: utf-8 -*-
import os
import torch
import torchaudio
import demucs
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio, convert_audio_channels
#https://github.com/facebookresearch/demucs

class Demucs:
    def __init__(self, name='htdemucs', device='cpu'):
        strpath='./models/demucs'
        strpath = os.path.join(os.path.dirname(__file__), strpath)
        self.model = get_model(name=name, repo=Path(strpath))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
    def proc(self, wav, sr=16000):
        model = self.model
        wav = torch.from_numpy(wav)[None]
        audio = convert_audio(wav, sr, 
                         model.samplerate, model.audio_channels)
        audio_dims = audio.dim()
        if audio_dims == 1:
            audio = audio[None, None].repeat_interleave(2, -2)
        else:
            if audio.shape[-2] == 1:
                audio = audio.repeat_interleave(2, -2)
            if audio_dims < 3:
                audio = audio[None]
        result = apply_model(model, audio, 
                             device=self.device, split=True, overlap=.25)        
        temp = result[0, model.sources.index('vocals')].mean(0)     
        temp = torchaudio.functional.resample(temp, model.samplerate, sr)
        return temp
