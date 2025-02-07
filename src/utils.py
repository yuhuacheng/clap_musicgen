import os
import torch
import requests
import torchaudio

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from dataclasses import dataclass
from tqdm import tqdm

def download_mp3(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # create a new directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)        
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download MP3 file: {url}, reason: {e}")
        return False

genres_to_include = [ # let's start with top genres
    "electronic",
    "alternative",
    "world",
    "hip-hop/rap",
    "r&b/soul",
    "pop",
    "rock",
    "dance/edm",
    "classical",
    "jazz",
    "blues",
    "country",
    "reggae",
    # TODO: african, latino...
]

def get_audio_file_path(id, top_genres):
    if top_genres and (len(top_genres) == 1) and (top_genres[0] in genres_to_include):
        shard = top_genres[0]
        return os.path.join(shard, f"{id}.mp3")    
    return os.path.join('other', f"{id}.mp3")    


def download_mp3_helper(row, audio_dir):
    """
    Helper function to:
      1. Calculate audio_path from the row.
      2. Download the file if it doesn't exist.
    """
    id = row['id']
    top_genres = row['top_genres']
    audio_url = row['audio_url']
    audio_path = os.path.join(audio_dir, get_audio_file_path(id, top_genres))
    
    if not os.path.exists(audio_path):
        if download_mp3(audio_url, audio_path):
            return audio_path
    return None

def parallel_download(train_data, audio_dir, max_workers=10):
    futures = []
    download_func = partial(download_mp3_helper, audio_dir=audio_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:        
        for _, row in train_data.iterrows():
            futures.append(executor.submit(download_func, row))
                
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()  # This will raise an exception if any
            except Exception as e:
                print(f"Error: {e}")
    print("All downloads complete!")


@dataclass
class TrainingSample:
    id: str
    caption: str
    waveform: torch.Tensor
    split: str

def load_single_audio(row, audio_dir, ap):
    """
    Loads the audio file for a single row, processes into 30s chunks,
    and returns a list of TrainingSample objects.
    """
    samples = []
    id = row['id']
    top_genres = row['top_genres']
    caption = row['caption']
    split = row['split']
    audio_path = os.path.join(audio_dir, get_audio_file_path(id, top_genres))

    try:
        waveform, sr = torchaudio.load(audio_path)
        # Perform your preprocessing (resampling, chunking, etc.)
        waveform = ap.run(waveform, sr)

        # Each row in waveform is a N sec chunk
        for i in range(waveform.shape[0]):
            chunk = waveform[i]
            samples.append(TrainingSample(id=id, caption=caption, waveform=chunk, split=split))

    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
    
    return samples

def parallel_load_audio(train_data, audio_dir, ap, max_workers=8):
    """
    Load & process all audio files in parallel using threads.
    Returns a single list of TrainingSample objects.
    """
    train_dataset_list = []

    # Create a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Submit one job per row in train_data
        for _, row in train_data.iterrows():
            futures.append(
                executor.submit(load_single_audio, row, audio_dir, ap)
            )

        # Collect results as they're completed
        for f in tqdm(as_completed(futures), total=len(futures)):
            # Each future returns a list of TrainingSample objects
            result_list = f.result()  # may raise if there was an error
            train_dataset_list.extend(result_list)

    return train_dataset_list
