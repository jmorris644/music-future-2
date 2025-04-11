from pydub import AudioSegment
from pydub.utils import which
AudioSegment.converter = which("ffmpeg")
from pydub import AudioSegment
import os

AudioSegment.converter = r"C:\\program files\\ffmpeg\\bin\\ffmpeg.exe"

input_dir = "./raw_music"
output_dir = "./converted_music"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith((".mp3", ".wav", ".m4a", ".aac")):
        try:
            print(f"🔁 Converting {file}...")
            audio = AudioSegment.from_file(os.path.join(input_dir, file))

            # Boost audio if it's too quiet (target -3 dBFS)
            if audio.dBFS < -12:
                gain = -3 - audio.dBFS
                print(f"   🔊 Applying gain: {gain:.2f} dB")
                audio = audio.apply_gain(gain)

            # Export to WAV
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".wav")
            audio.export(output_path, format="wav")
        except Exception as e:
            print(f"❌ Failed to convert {file}: {e}")
