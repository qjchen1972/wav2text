import time   
from wav2text import Wav2text, text2pinyin, wav2array

if __name__ == "__main__":
    
    device='cpu'
    pq = Wav2text(device=device)
    start = time.time()
    res = pq.wav2text('./wav/ss.wav')
    print(time.time() - start, res)