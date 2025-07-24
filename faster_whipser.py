
from faster_whisper import WhisperModel

model = WhisperModel("large-v3")
audios = ["1.wav", "2.wav","3.wav","4.wav","5.wav","6.wav"]
for audio in audios:
    segments, info = model.transcribe(audio)
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
