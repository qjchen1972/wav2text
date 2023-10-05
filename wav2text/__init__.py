# -*- coding: utf-8 -*-

#https://github.com/openai/whisper
#sudo apt update && sudo apt install ffmpeg
#pip install -U openai-whisper
import os
import whisper
import zhconv
from .util import text2pinyin, wav2array, load_audio,save_audio
from .demucs import Demucs

class Wav2text:
    
    def __init__(self, name='base',
                       download_root='./models/whisper',
                       device='cpu', #'cuda'
                       no_speech_prob=0.6,
                       avg_logprob=-3):
        
        download_root = os.path.join(os.path.dirname(__file__), download_root)
        self.model = whisper.load_model(name=name, 
                               device=device,
                               download_root=download_root,
                               in_memory=True)
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob                            
    
    def wav2text(self, wavpath=None):      
        if wavpath is None:
            return []    
        result = self.model.transcribe(
                            wavpath, 
                            temperature=0, 
                            language='zh',) 
        res = []
        for v in result['segments']:        
            if (result['language'] == 'zh' and 
               v['no_speech_prob'] < self.no_speech_prob and 
               v['avg_logprob'] > self.avg_logprob):
                
                text = zhconv.convert(v['text'], 'zh-cn')
                pinyin = text2pinyin(text)
                res.append({'text': text, 'pinyin': pinyin})
        
        if not res:
            text = '我听不懂你说什么'
            pinyin = text2pinyin(text)
            res.append({'text': text, 'pinyin': pinyin})           
        return res


    
    
