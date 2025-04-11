import os
import requests
from tqdm import tqdm
from pydub import AudioSegment
from io import BytesIO

# Output directory for raw audio
raw_music_dir = "./raw_music"
os.makedirs(raw_music_dir, exist_ok=True)

# List of 5+ Creative Commons music sample links
music_urls = [
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Lobo_Loco/Salad_Mixed/Salad_Mixed_03_Beat_Like_this_ID_1173.mp3",
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Komiku/Chill_Adventure/Komiku_-_09_-_Rainy_Day_Games.mp3",
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Scott_Holmes_Music/Happy_Music/Scott_Holmes_Music_-_03_-_Upbeat_Party.mp3",
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Ars_Sonor/Creeping_Through_the_Night/Ars_Sonor_-_03_-_Hollow.mp3",
    "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Podington_Bear/Happy_Happy/Podington_Bear_-_Starling.mp3"
]

for i, url in enumerate(tqdm(music_urls, desc="Downloading music")):
    try:
        response = requests.get(url, timeout=10)
        audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
        audio.export(os.path.join(raw_music_dir, f"track_{i+1}.mp3"), format="mp3")
    except Exception as e:
        print(f"❌ Failed to download or convert {url}: {e}")

print("✅ Download complete. Files saved to ./raw_music")
