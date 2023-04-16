"""Example of using the Whisper library to detect the language and transcribe."""
import os
import sys
import time
from typing import Optional

import torch
import whisper
from pytube import YouTube

if not torch.cuda.is_available():
    print("no cuda")
    sys.exit()


def download_video(url: str) -> Optional[str]:
    """Download a YouTube video."""
    try:
        yt_dl = YouTube(url)
        stream = yt_dl.streams.filter(only_audio=True).first()
        if stream is None:
            print("No audio stream found")
            return None
        return stream.download(output_path=os.path.join(os.getcwd(), "outputs"))
    except Exception as err:  # pylint: disable=broad-except
        print(f"Error downloading video: {err}")
        return None


devices = torch.device("cuda:0")  # pylint: disable=no-member
MODEL = whisper.load_model("tiny.en", device=devices)

downloaded_file = download_video("https://www.youtube.com/watch?v=GwzN5YwMzv0")
if downloaded_file is None:
    sys.exit(2)

# load audio and pad/trim it to fit full audio frames
audio = whisper.load_audio(downloaded_file)
# audio = whisper.pad_or_trim(audio, 300 * model.sample_rate)

# make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)


options = {
    "language": "en",
    "task": "transcribe",
}


start_time = time.time()
result = whisper.transcribe(MODEL, audio, **options)
print(result["text"])

end_time = time.time()
execution_time = end_time - start_time
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(
    "Transcription elapsed execution time:"
    f" {int(hours)}:{int(minutes):02d}:{seconds:06.3f}"
)
