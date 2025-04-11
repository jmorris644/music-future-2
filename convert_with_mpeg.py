import os
import subprocess

input_dir = "./raw_music"
output_dir = "./converted_music"
os.makedirs(output_dir, exist_ok=True)

ffmpeg_path = r"C:\program files\ffmpeg\bin\ffmpeg.exe"  # Update path if needed

for file in os.listdir(input_dir):
    if file.lower().endswith(".mp3"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".wav")

        print(f"🎵 Converting {file}...")
        try:
            subprocess.run([
                ffmpeg_path, "-y", "-i", input_path,
                "-filter:a", "volume=5.0",
                "-ar", "11025",
                "-ac", "1",
                "-sample_fmt", "s16",
                output_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed for {file}: {e}")
