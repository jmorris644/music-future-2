from pydub import AudioSegment
import os

input_dir = "./raw_music"
output_dir = "./converted_music"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith((".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a")):
        try:
            print(f"Converting {file}...")
            audio = AudioSegment.from_file(os.path.join(input_dir, file))
            audio.export(os.path.join(output_dir, os.path.splitext(file)[0] + ".wav"), format="wav")
        except Exception as e:
            print(f"❌ Could not convert {file}: {e}")
