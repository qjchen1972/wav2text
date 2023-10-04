# -*- coding: utf-8 -*-
import os
import numpy as np
from io import BytesIO
import ffmpeg
from xpinyin import Pinyin

def text2pinyin(text):   
    res = Pinyin().get_pinyin(text)
    return res.replace('-', ' ')
    
#音频流转np.array    
def wav2array(audio, sr=16000):

    bytes_io_file = BytesIO(audio.stream.read())
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
        