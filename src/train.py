import os
import time
import json 

from dataclasses import dataclass
from datetime import datetime

import numpy as np 
import torch 
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

from src.eval import compute_audio_embeddings, eval_track_similarity, calculate_ndcg, genre_mapping


def compute_loss(audio_proj, text_proj, pretrained_text_embs, logit_scale_a, logit_scale_t):
    """
    Computes bidirectional ListMLE loss for full ranking optimization:
    - Audio-to-Text similarity ranking loss
    - Text-to-Audio similarity ranking loss
    """

    # # Compute predicted pairwise similarities
    sim_matrix_at = torch.matmul(audio_proj, text_proj.T) * logit_scale_a.exp()
    sim_matrix_ta = torch.matmul(text_proj, audio_proj.T) * logit_scale_t.exp()

    # Normalize similarities to prevent extreme values
    def normalize_similarity(sim_matrix):
        sim_matrix = sim_matrix - sim_matrix.mean(dim=1, keepdim=True)
        return sim_matrix / (sim_matrix.std(dim=1, keepdim=True) + 1e-6)

    sim_matrix_at = normalize_similarity(sim_matrix_at)
    sim_matrix_ta = normalize_similarity(sim_matrix_ta)

    # Normalize pretrained text embeddings and compute text similarity ground-truth
    pretrained_text_embs = F.normalize(pretrained_text_embs, p=2, dim=-1)
    pretrained_text_sim = torch.matmul(pretrained_text_embs, pretrained_text_embs.T)  # Ground-truth similarity

    # Get sorted indices based on ground-truth text similarity ranking (descending order)
    sorted_indices = torch.argsort(pretrained_text_sim, dim=1, descending=True)  # (batch_size, batch_size)

    # Gather predicted similarities based on ground-truth order
    sorted_sim_matrix_at = torch.gather(sim_matrix_at, dim=1, index=sorted_indices)  # Audio-to-Text
    sorted_sim_matrix_ta = torch.gather(sim_matrix_ta, dim=1, index=sorted_indices)  # Text-to-Audio

    # Compute ListMLE loss for both directions
    def compute_listmle_loss(sorted_sim_matrix):
        cumsum_log_probs = []
        for i in range(sorted_sim_matrix.shape[1]):
            probs = F.log_softmax(sorted_sim_matrix[:, i:], dim=1)[:, 0]
            cumsum_log_probs.append(probs)

        return -torch.sum(torch.stack(cumsum_log_probs, dim=1), dim=1).mean()

    loss_at = compute_listmle_loss(sorted_sim_matrix_at)  # Audio-to-Text loss
    loss_ta = compute_listmle_loss(sorted_sim_matrix_ta)  # Text-to-Audio loss

    # Combine both losses
    total_loss = loss_at + loss_ta
    return total_loss


def validate(model, dataloader, device, dtype):
    """Compute validation loss."""
    model.eval()  # Set the model to evaluation mode
    val_loss = 0

    with torch.no_grad():  # Disable gradient computation for validation
        pbar = tqdm(dataloader, desc="Validating", dynamic_ncols=True, leave=True, position=0)
        for step, (ids, audio_waveforms, captions) in enumerate(pbar):                                    

            audio_waveforms = audio_waveforms.to(device)
            captions = {k: v.to(device) for k, v in captions.items()}

            with torch.autocast(device_type=device, dtype=dtype):
                # Forward pass
                audio_proj, text_proj, pretrained_text_embs, logit_scale_a, logit_scale_t = model(ids, audio_waveforms, captions)            
                loss = compute_loss(audio_proj, text_proj, pretrained_text_embs, logit_scale_a, logit_scale_t)                
                val_loss += loss.item()

    return val_loss / len(dataloader)


def evaluate(model, eval_ids, all_dataset_list, device, audio_metadata, top_k=5):
    print('evaluating on a few tracks...')
    model.eval()
    audio_embeddings = compute_audio_embeddings(all_dataset_list, model, device)
    sim_results, logs = eval_track_similarity(
        eval_ids=eval_ids,
        audio_embeddings=audio_embeddings,
        top_k=top_k,
        audio_metadata=audio_metadata,
    )
    emb_std = torch.cat(list(audio_embeddings.values())).view(-1).std().item()
    ndcg = calculate_ndcg(sim_results, genre_mapping, top_k=top_k)
    return ndcg, logs, emb_std


def update_plot(train_losses, val_losses, eval_ndcgs, eval_emb_stds, epoch):    
    """Override the training and validation loss plot with a secondary y-axis."""
    from matplotlib.ticker import MaxNLocator

    def exponential_moving_avg(data, alpha=0.1):
        """Applies exponential moving average (EMA) smoothing."""
        smoothed = []
        if not data:
            return smoothed
        smoothed.append(data[0])  # First value remains the same
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i - 1])
        return smoothed

    # Apply smoothing to train and validation losses
    train_losses_smoothed = exponential_moving_avg([np.log(loss.metric + 1) for loss in train_losses])
    val_losses_smoothed = exponential_moving_avg([np.log(loss.metric + 1) for loss in val_losses])

    clear_output(wait=True)
    
    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Adjust x-values so they correctly track actual steps instead of going negative
    train_steps = [loss.step for loss in train_losses]
    val_steps = [loss.step for loss in val_losses]
    ndcg_steps = [ndcg.step for ndcg in eval_ndcgs]
    emb_std_steps = [emb_std.step for emb_std in eval_emb_stds]

    eval_ndcgs = [ndcg.metric for ndcg in eval_ndcgs]
    eval_emb_stds = [emb_std.metric for emb_std in eval_emb_stds]
    
    # Plot training loss on the primary y-axis
    ax1.plot(
        train_steps,
        train_losses_smoothed,
        label="Training Loss",
        color="blue",
        alpha=0.75
    )
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("log(Training Loss)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks for better readability
    ax1.grid(True, alpha=0.5)

    # Create a secondary y-axis for validation loss
    ax2 = ax1.twinx()
    ax2.plot(
        val_steps,
        val_losses_smoothed,
        label="Validation Loss",
        color="orange",
        linestyle="--",
        marker="o"
    )
    ax2.set_ylabel("Log(Validation Loss)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks for better readability

    # Create a third y-axis for eval_ndcg
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
    ax3.plot(
        ndcg_steps,
        eval_ndcgs,
        label="Eval NDCG",
        color="green",
        linestyle="dotted",
        marker="s"
    )
    ax3.set_ylabel("Eval NDCG", color="green")
    ax3.tick_params(axis="y", labelcolor="green")
    ax3.set_ylim(0.0, max(eval_ndcgs) * 1.1)  # Set the y-axis to start at 0.0 and adjust upper limit for padding
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks for better readability

    # Create a forth y-axis for eval_emb_stds
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("outward", 120))  # Offset the fourth axis further out
    ax4.plot(
        emb_std_steps,
        eval_emb_stds,
        label="Eval Embedding Stds",
        color="red",
        linestyle="dashdot",
        marker="^"
    )
    ax4.set_ylabel("Eval Embedding Stds", color="red")
    ax4.tick_params(axis="y", labelcolor="red")
    ax4.set_ylim(min(eval_emb_stds) * 0.9, max(eval_emb_stds) * 1.1)  # Scale range with padding
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks for better readability

    # Add title and legends
    plt.title(f"Training and Validation Loss with Eval NDCG (Epoch {epoch})")
    # put legend to the lower left part of the figure
    fig.legend(loc="lower left", bbox_to_anchor=(0.1, 0.1))
    # fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))

    # Show the plot
    plt.show()

@dataclass
class MetricAndStep:
    metric: float
    step: int

def train_epochs(
        train_dataloader, 
        val_dataloader,
        eval_ids,
        train_losses,
        val_losses,
        eval_ndcgs,
        eval_emb_stds,
        curr_epoch, 
        epochs, 
        model, 
        optimizer, 
        scaler, 
        device,         
        audio_metadata,
        all_dataset_list,
        model_save_path,        
        best_val_loss = None,
        best_eval_ndcg = None,
        top_k=5,
        log_every_n_steps=200,        
        ):
    
    dtype = torch.float16 if device != "mps" else torch.float32
    best_val_loss = float('inf') if best_val_loss is None else best_val_loss
    best_eval_ndcg = float('-inf') if best_eval_ndcg is None else best_eval_ndcg    
    global_steps = 0
    
    def log_metric(epoch, train_loss, best_eval_ndcg, step):
        # Validation step
        val_loss = validate(model, val_dataloader, device, dtype)        
        val_losses.append(MetricAndStep(val_loss, step))  # Log validation loss for this epoch

        # Evaluate on a few tracks
        eval_ndcg, eval_logs, eval_emb_std = evaluate(model=model, eval_ids=eval_ids, all_dataset_list=all_dataset_list, device=device, audio_metadata=audio_metadata, top_k=top_k)
        eval_ndcgs.append(MetricAndStep(eval_ndcg, step))
        eval_emb_stds.append(MetricAndStep(eval_emb_std, step))

        # Update plot dynamically        
        update_plot(train_losses, val_losses, eval_ndcgs, eval_emb_stds, epoch=epoch)
        
        print(f"Epoch {epoch}, Train Loss: {train_loss}")
        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch}, Eval NDCG: {eval_ndcg:.4f}")

        # print eval_logs list line by line
        print(f"Epoch {epoch}, Evaluation Logs")        
        for log in eval_logs:
            print(log)

        if (model_save_path is not None) and (eval_ndcg > best_eval_ndcg):
            best_eval_ndcg = eval_ndcg
            # print(f"New best validation loss: {val_loss:.4f}. Saving model...")
            print(f"New best eval ndcg: {eval_ndcg:.4f}. Saving model...")
            model.save_pretrained(model_save_path)

            # Save metadata
            metadata = {
                "epoch": epoch,
                "validation_loss": val_loss,
                "train_loss": train_loss,
                "eval_ndcg": eval_ndcg,
            }
            metadata_path = f"{model_save_path}/metadata.json"
            with open(metadata_path, "w") as metadata_file:
                json.dump(metadata, metadata_file, indent=4)            

        return val_loss, eval_ndcg, eval_logs, best_eval_ndcg
 

    for epoch in range(curr_epoch, curr_epoch + epochs):        
        # Training loop
        model.train()
        epoch_loss = 0

        if epoch == 0:
            log_metric(epoch, train_loss=None, best_eval_ndcg=best_eval_ndcg, step=0)

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{curr_epoch+epochs-1}", dynamic_ncols=True, leave=True, position=0)
        for step, (ids, audio_waveforms, captions) in enumerate(pbar):
            # Map string IDs to integers for positive mask calculation
            audio_waveforms = audio_waveforms.to(device)
            captions = {k: v.to(device) for k, v in captions.items()}

            start_time = time.time()

            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=dtype):
                # Forward pass            
                audio_proj, text_proj, pretrained_text_embs, logit_scale_a, logit_scale_t = model(ids, audio_waveforms, captions)                
                loss = compute_loss(audio_proj, text_proj, pretrained_text_embs, logit_scale_a, logit_scale_t)
                                         
                train_losses.append(MetricAndStep(loss.item(), global_steps))
                epoch_loss += loss.item()

            # -----------------------
            # Backward + Optim Step
            # -----------------------
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize() if device == 'cuda' else None # make sure GPU operations are done before measuring time

            elapsed_time = time.time() - start_time
            processed_audio_samples = audio_waveforms.numel()
            processed_text_tokens = captions["input_ids"].numel()
            audio_throughput = processed_audio_samples / elapsed_time
            text_throughput = processed_text_tokens / elapsed_time

            # Update tqdm progress bar postfix for real-time info
            pbar.set_postfix(
              step=step + 1,
              loss=f"{loss.item():.4f}",
              waveform_samples_per_sec=f"{audio_throughput:.2f}",
              text_tokens_per_sec=f"{text_throughput:.2f}"
            )

            global_steps += 1

            # every N steps, log the metrics
            if (global_steps != 0) and (global_steps % log_every_n_steps == 0):
                train_loss = epoch_loss/step
                _, _, _, new_best_eval_ndcg = log_metric(epoch, train_loss, best_eval_ndcg, global_steps)
                best_eval_ndcg = new_best_eval_ndcg            

        pbar.close()        
        
    return epoch, best_val_loss, best_eval_ndcg