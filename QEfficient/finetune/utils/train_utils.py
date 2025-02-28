# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import torch_qaic  # noqa: F401
    import torch_qaic.debug as qaic_debug  # noqa: F401
    import torch_qaic.profile as qaic_profile  # noqa: F401
except ImportError as e:
    print(f"Warning: {e}. Moving ahead without these qaic modules.")

from ..configs import TrainConfig


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config: TrainConfig,
    device,
    local_rank=None,
    rank=None,
):
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []
    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = (
            f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False
    tensorboard_updates = SummaryWriter() if not train_config.enable_ddp or local_rank == 0 else None
    scaler = GradScaler() if train_config.grad_scaler else None
    loss_0_counter = torch.tensor([0]).to(device)
    if train_config.enable_ddp:
        dist.broadcast(loss_0_counter, src=0)

    for epoch in range(train_config.num_epochs):
        if loss_0_counter.item() == train_config.convergence_counter:
            if train_config.enable_ddp:
                print(
                    f"Not proceeding with epoch {epoch + 1} on device {local_rank} since loss <= {train_config.convergence_loss} for {loss_0_counter.item()} steps."
                )
            else:
                print(
                    f"Not proceeding with epoch {epoch + 1} since loss <= {train_config.convergence_loss} for {loss_0_counter.item()} steps."
                )
            break
        print(f"Starting epoch {epoch + 1}/{train_config.num_epochs}")
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_length = len(train_dataloader) // gradient_accumulation_steps
        pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch + 1}", total=total_length, dynamic_ncols=True)
        qaic_profile.start_profiling(device, 1) if train_config.use_profiler and "torch_qaic" in globals() else None

        for step, batch in enumerate(train_dataloader):
            total_train_steps += 1
            if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                max_steps_reached = True
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with (
                torch.autocast(device_type=device, dtype=torch.float16) if train_config.use_autocast else nullcontext()
            ):
                if train_config.opByOpVerifier and "qaic_debug" in globals():
                    with qaic_debug.OpByOpVerifierMode(
                        ref_device="cpu",
                        ref_dtype=torch.float32,
                        atol=1e-1,
                        use_ref_output_on_mismatch=True,
                        max_failures=None,
                        repeat_same_op=True,
                        dump_root_dir=f"{train_config.dump_root_dir}{step}",
                    ) as verifier:
                        loss = model(**batch).loss
                    print("Mismatches detected:", verifier.get_perop_mismatch_count())
                else:
                    loss = model(**batch).loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            if train_config.enable_ddp:
                if local_rank == 0:
                    loss_0_counter = torch.tensor(
                        [loss_0_counter.item() + 1 if loss <= train_config.convergence_loss else 0]
                    ).to(device)
                dist.broadcast(loss_0_counter, src=0)
            else:
                loss_0_counter = torch.tensor(
                    [loss_0_counter.item() + 1 if loss <= train_config.convergence_loss else 0]
                ).to(device)
            if tensorboard_updates:
                tensorboard_updates.add_scalars("loss", {"train": loss}, total_train_steps)
            if train_config.save_metrics:
                train_step_loss.append(loss.detach().float().item())
                train_step_perplexity.append(float(torch.exp(loss.detach().float())))
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)
            if step % train_config.intermediate_step_save == 0:
                qaic_profile.stop_profiling(
                    device
                ) if train_config.use_profiler and "qaic_profile" in globals() else None
                if train_config.enable_ddp and dist.get_rank() == 0:
                    model.module.save_pretrained(f"{train_config.output_dir}/trained_weights/step_{step}")
                elif not train_config.enable_ddp:
                    model.save_pretrained(f"{train_config.output_dir}/trained_weights/step_{step}")
            pbar.set_description(
                f"Training Epoch: {epoch + 1}/{train_config.num_epochs}, step {step + 1}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
            )
            if train_config.save_metrics:
                save_to_json(
                    metrics_filename,
                    train_step_loss,
                    train_loss,
                    train_step_perplexity,
                    train_prep,
                    val_step_loss,
                    val_loss,
                    val_step_perplexity,
                    val_prep,
                )
            if loss_0_counter.item() == train_config.convergence_counter:
                print(
                    f"Loss <= {train_config.convergence_loss} for {loss_0_counter.item()} steps. Stopping fine-tuning{' on device ' + str(local_rank) if train_config.enable_ddp else ''}."
                )
                break
        pbar.close()
        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        train_epoch_loss = total_loss / (
            step if loss_0_counter.item() == train_config.convergence_counter else len(train_dataloader)
        )
        train_perplexity = torch.exp(train_epoch_loss)
        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))
        lr_scheduler.step()
        should_save_model = train_config.save_model

        if train_config.run_validation:
            if train_config.enable_ddp:
                dist.barrier()
                eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                    model, train_config, eval_dataloader, local_rank, tokenizer, device
                )
                dist.barrier()
                dist.all_reduce(eval_epoch_loss, op=dist.ReduceOp.SUM)
                if local_rank == 0 and tensorboard_updates:
                    tensorboard_updates.add_scalars("loss", {"eval": eval_epoch_loss}, total_train_steps)
            else:
                eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                    model, train_config, eval_dataloader, local_rank, tokenizer, device
                )
                if tensorboard_updates:
                    tensorboard_updates.add_scalars("loss", {"eval": eval_epoch_loss}, total_train_steps)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            should_save_model = train_config.save_model and eval_epoch_loss < best_val_loss
        if should_save_model:
            if train_config.enable_ddp and dist.get_rank() == 0:
                model.module.save_pretrained(train_config.output_dir)
            elif not train_config.enable_ddp:
                model.save_pretrained(train_config.output_dir)
        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"Best eval loss on epoch {epoch + 1}: {best_val_loss}")
            val_loss.append(float(eval_epoch_loss))
            val_prep.append(float(eval_ppl))
        print(
            f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
        )
        if train_config.save_metrics:
            save_to_json(
                metrics_filename,
                train_step_loss,
                train_loss,
                train_step_perplexity,
                train_prep,
                val_step_loss,
                val_loss,
                val_step_perplexity,
                val_prep,
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if checkpoint_times else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)
    results.update(
        {
            "avg_train_prep": avg_train_prep,
            "avg_train_loss": avg_train_loss,
            "avg_epoch_time": avg_epoch_time,
            "avg_checkpoint_time": avg_checkpoint_time,
        }
    )
    if train_config.run_validation:
        results.update({"avg_eval_prep": avg_eval_prep, "avg_eval_loss": avg_eval_loss})
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, device):
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0
    total_eval_steps = 0
    for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="Evaluating Epoch", dynamic_ncols=True)):
        total_eval_steps += 1
        if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            with (
                torch.autocast(device_type=device, dtype=torch.float16) if train_config.use_autocast else nullcontext()
            ):
                outputs = model(**batch)
            loss = outputs.loss
            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))
            eval_loss += loss.detach().float()
        preds = torch.argmax(outputs.logits, -1)
        eval_preds.extend(tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True))
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    print(f"eval_ppl={eval_ppl.detach().cpu()} eval_epoch_loss={eval_epoch_loss.detach().cpu()}")
    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def print_model_size(model, config) -> None:
    print(f"--> Model {config.model_name}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def save_to_json(
    output_filename,
    train_step_loss,
    train_epoch_loss,
    train_step_ppl,
    train_epoch_ppl,
    val_step_loss,
    val_epoch_loss,
    val_step_ppl,
    val_epoch_ppl,
):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
