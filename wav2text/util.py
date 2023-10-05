# -*- coding: utf-8 -*-
import os
import numpy as np
from io import BytesIO
import ffmpeg
import torchaudio
from xpinyin import Pinyin

def text2pinyin(text):   
    res = Pinyin().get_pinyin(text)
    return res.replace('-', ' ')
    
#字节流转np.array    
def wav2array(audio, sr=16000):
    bytes_io_file = BytesIO(audio) #.stream.read())
    bytes_io_file.seek(0)
    content = bytes_io_file.getvalue()
    process = (
            ffmpeg.input("pipe:")
            .output("pipe:", acodec="pcm_s16le", format="s16le", ac=1, ar=sr)            
            .run_async(pipe_stdin=True, pipe_stderr=True, pipe_stdout=True)
        )       
    try:
        buffer, _ = process.communicate(input=content)
        process.stdin.close()
    except Exception as e:
        print(f'wav2array is Error : {e}')
        return None    
    return np.frombuffer(buffer, np.int16).flatten().astype(np.float32) / 32768.0
        
def load_audio(file: str, sr: int = 16000):    
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def save_audio(path, data, sample_rate=16000):
    audio_dims = data.dim()
    if audio_dims == 1: data = data[None]
    torchaudio.save(path, data, sample_rate)