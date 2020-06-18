import wave


class Pcm2Wav(object):
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.bits = 16

    def __call__(self, pcm_path, wav_path):
        pcm = None

        with open(pcm_path, 'rb') as pcmdata:
            pcm = pcmdata.read()

        with wave.open(wav_path, 'wb') as wavdata:
            wavdata.setnchannels(self.channels)
            wavdata.setsampwidth(self.bits // 8)
            wavdata.setframerate(self.sample_rate)
            wavdata.writeframes(pcm)