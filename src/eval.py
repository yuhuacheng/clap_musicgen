import numpy as np

import os
import torch

from typing import List, Dict

import torch.nn.functional as F

from src.utils import TrainingSample
from src.modules.clap_model import CLAPModel

# True genre mapping, manually selected / validated
genre_mapping = {
    "00d2438d-0ae9-479c-a63a-ed78a2c44368": "heavy metal",
    "01a72954-0cf4-4fe0-bb8e-cb299874a34e": "heavy metal",
    "00831752-ac48-4201-afc8-d15d2ab3b2fc": "country rock (soft)",
    "011fc464-80c3-4b18-b930-7bef81242b10": "country rock (soft)",
    "0192198f-59b0-4fab-978d-7c0b68320cb2": "fast contemporary pop",
    "023c8c04-5ba3-42a9-aea2-904416971188": "fast contemporary pop",
    "04dcb79c-dd33-4b3d-827c-b47e97b6373b": "fast contemporary pop",
    "09cb51d6-8f5d-463c-bf99-dc07353b55cf": "fast contemporary pop",
    "0722191d-e03b-469e-b1fe-2f664fe5a3d1": "hip hop /rap",
    "0b821192-1acc-4d92-b9c0-07035f3c7092": "hip hop /rap",
    "0cac4971-75a1-4f87-9078-3136e5a014d9": "hip hop /rap",
    "16030b83-61d3-41db-b6b7-dc3337733d97": "funk / soul",
    "08e51a9b-5e87-4199-835e-1b1f93afccd4": "funk / soul",
    "1884ea6c-6ae7-4be6-8953-67b7ce9cb12b": "funk / soul",
    "c7e33f64-fe71-4c18-a71f-47c38ba17de2": "funk / soul",
    "03008bd5-6901-4fbc-a9d4-56b76733daf0": "slow contemporary",
    "18721de6-c791-48d5-b34c-13db21c5883f": "slow contemporary",
    "07b3416e-df92-4b41-a7ca-e2bb5eaa4df4": "classical / orchestral",
    "074ae05c-112c-4787-894b-16df879f4cb9": "classical / orchestral",
    "0a4b6d10-2c82-4937-a31a-8c9158236dad": "lofi jazz",
    "6488fd64-b959-45bc-9f0f-6f6e107c6735": "lofi jazz",
    "0cf0fafb-84ac-43ab-a68b-22fb6c923b65": "electronic / dance",
    "0d7cfeb6-8078-47d1-a2d5-f17caa6f631c": "electronic / dance",
    "0ee29c2e-d3e6-40c0-b64d-8bcba32562a3": "electronic / dance",
}

def compute_audio_embeddings(dataset_list: List[TrainingSample], model: CLAPModel, device: str) -> Dict[str, torch.Tensor]:
    audio_embeddings = {}
    model.eval()

    for d in dataset_list:
        track_id = d.id
        waveform = d.waveform.unsqueeze(0)  # Add batch dimension
        waveform = waveform.to(device)
        
        with torch.no_grad():
            audio_embedding = model.audio_encoder([track_id], waveform) # wrap track_id to list since it expects a batch
        
        # audio_embedding is a tensor with shape (1, projection_dim), but since each track_id has multiple chunks, we will simply stack them into one tensor with shape of (B, projection_dim)
        if track_id in audio_embeddings:
            audio_embeddings[track_id] = torch.cat([audio_embeddings[track_id], audio_embedding])
        else:
            audio_embeddings[track_id] = audio_embedding        
        
    return audio_embeddings

def eval_track_similarity(eval_ids: List[str], audio_embeddings: Dict[str, torch.Tensor], audio_metadata: Dict[str, str], top_k: int):
    def get_genres(track_id):
        return (track_id.split('-')[-1], audio_metadata.get(track_id))
    
    # for each track in eval_ids, find the top k similar track in audio_embeddings
    # return the track_id and cosine similarity
    results = {}
    logs = []
    for eval_id in eval_ids:
        eval_emb = audio_embeddings[eval_id]
        with torch.no_grad():
            eval_emb_norm = F.normalize(eval_emb, p=2, dim=1)
            sim = {}
            for track_id, emb in audio_embeddings.items():
                if track_id != eval_id:                                           
                    emb_norm = F.normalize(emb, p=2, dim=1)

                    # pairwise cosine similarity
                    cosine_similarity_mat = torch.mm(eval_emb_norm, emb_norm.t())                
                    sim[track_id] = cosine_similarity_mat.mean().item()
    
        sim = dict(sorted(sim.items(), key=lambda item: item[1], reverse=True))
        sim = {k: sim[k] for k in list(sim)[:top_k]}

        logs.append(f"eval track: {get_genres(eval_id)}, top k:: {[get_genres(_) for _ in list(sim.keys())]}")
        results[eval_id] = list(sim.keys())

    return results, logs

def calculate_ndcg(recommendations, genre_mapping, top_k):
    """
    Calculate the average NDCG for a set of recommendations.

    Parameters:
        recommendations (dict): A dictionary where each key is a track ID and the value is a list of recommended track IDs.
        genre_mapping (dict): A dictionary mapping track IDs to their genres.
        top_k (int): The number of recommendations to consider for NDCG calculation.

    Returns:
        float: The average NDCG score across all track IDs.
    """
    def ndcg_at_k(relevant_scores, k):
        """Calculate NDCG for a single track."""
        dcg = 0.0
        for i, rel in enumerate(relevant_scores[:k]):
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

        ideal_scores = sorted(relevant_scores, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_scores[:k]):
            idcg += rel / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    # Create a mapping from track ID to genre
    avg_ndcg = []

    for track_id, predicted_ids in recommendations.items():
        if track_id not in genre_mapping:
            raise ValueError(f"Track ID {track_id} not found in genre mapping.")

        true_genre = genre_mapping[track_id]

        # Determine relevance scores based on the same genre
        relevance_scores = [1 if genre_mapping.get(pred_id) == true_genre else 0 for pred_id in predicted_ids]

        # Calculate NDCG for the current track
        ndcg = ndcg_at_k(relevance_scores, top_k)
        avg_ndcg.append(ndcg)

    return np.mean(avg_ndcg) if avg_ndcg else 0.0

