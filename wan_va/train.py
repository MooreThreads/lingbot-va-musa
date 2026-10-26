# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import copy
import os
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.musa_bootstrap import bootstrap_musa

bootstrap_musa()

import wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json
import shutil

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)
from utils.device import (
    default_attn_mode,
    empty_cache,
    get_torch_device,
    is_musa,
    synchronize,
)

from dataset import MultiLatentLeRobotDataset, collate_fixed_latent_frames
import gc


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name='test_lln'
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = get_torch_device(config.local_rank)
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode=getattr(config, "attn_mode", default_attn_mode("flex"))
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
        parameters = [p for p in self.transformer.parameters() if p.requires_grad]
        if is_musa():
            from torch_musa.optim import FusedAdamW

            optimizer_cls = FusedAdamW
        else:
            optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(
            config=config,
            num_init_worker=getattr(config, "dataset_init_worker", 128),
        )
        if len(train_dataset) == 0:
            raise ValueError(f"No training samples found under {config.dataset_path}")
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=(
                partial(collate_fixed_latent_frames, target_frames=config.train_latent_frames)
                if config.get('train_latent_frames') is not None
                else None
            ),
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        if not torch.isfinite(latent_loss) or not torch.isfinite(action_loss):
            raise FloatingPointError(
                f"Non-finite loss at step {self.step}: "
                f"latent_loss={latent_loss.detach().float().item()}, "
                f"action_loss={action_loss.detach().float().item()}"
            )
        loss = latent_loss + action_loss

        loss.backward()

        losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        
        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            if not torch.isfinite(total_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at step {self.step}: "
                    f"total_norm={total_norm.detach().float().item()}"
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self,):
        """Gather and atomically publish a validated transformer checkpoint."""
        state_dict = {}
        gather_error = None
        if dist.is_initialized():
            dist.barrier()
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        except Exception as exc:
            gather_error = exc

        if dist.is_initialized():
            gather_status = torch.tensor(
                [0 if gather_error is not None else 1],
                dtype=torch.int32,
                device=self.device,
            )
            dist.all_reduce(gather_status, op=dist.ReduceOp.MIN)
            if gather_status.item() == 0:
                raise RuntimeError("Checkpoint state gathering failed") from gather_error
        elif gather_error is not None:
            raise RuntimeError("Checkpoint state gathering failed") from gather_error

        save_error = None
        if self.config.rank == 0:
            checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
            temp_dir = self.save_dir / f".checkpoint_step_{self.step}.tmp-{os.getpid()}"
            try:
                if checkpoint_dir.exists():
                    raise FileExistsError(f"Checkpoint already exists: {checkpoint_dir}")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                transformer_dir = temp_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=False)
                logger.info("Saving transformer to %s", transformer_dir)

                # Keep individual safetensors shards bounded. Building one 10GB
                # BF16 dictionary duplicates the gathered full state in memory.
                shard_limit = 2 * 1024**3
                shard_index = 1
                shard_bytes = 0
                shard_state = {}
                weight_map = {}
                total_size = 0

                def flush_shard():
                    nonlocal shard_index, shard_bytes, shard_state
                    if not shard_state:
                        return
                    filename = (
                        f"diffusion_pytorch_model-{shard_index:05d}.of-00000.safetensors"
                    )
                    save_file(shard_state, transformer_dir / filename)
                    shard_index += 1
                    shard_bytes = 0
                    shard_state = {}

                for key, value in state_dict.items():
                    tensor = value.to(torch.bfloat16)
                    if not torch.isfinite(tensor).all():
                        raise RuntimeError(f"Non-finite tensor in checkpoint: {key}")
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    if shard_state and shard_bytes + tensor_bytes > shard_limit:
                        flush_shard()
                    shard_state[key] = tensor
                    weight_map[key] = (
                        f"diffusion_pytorch_model-{shard_index:05d}.of-00000.safetensors"
                    )
                    shard_bytes += tensor_bytes
                    total_size += tensor_bytes
                flush_shard()

                shard_count = shard_index - 1
                if shard_count == 1:
                    old_name = transformer_dir / "diffusion_pytorch_model-00001.of-00000.safetensors"
                    model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                    old_name.rename(model_file)
                    weight_map = {key: model_file.name for key in weight_map}
                else:
                    for path in transformer_dir.glob("diffusion_pytorch_model-*.of-00000.safetensors"):
                        path.rename(path.with_name(path.name.replace("of-00000", f"of-{shard_count:05d}")))
                    weight_map = {
                        key: name.replace("of-00000", f"of-{shard_count:05d}")
                        for key, name in weight_map.items()
                    }
                    with (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").open("w") as file:
                        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, file, indent=2)

                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with config_file.open('w') as file:
                    json.dump(config_dict, file, indent=2)
                    file.flush()
                    os.fsync(file.fileno())

                if not config_file.is_file() or not any(transformer_dir.glob("*.safetensors")):
                    raise RuntimeError("Checkpoint files were not created")
                with config_file.open() as file:
                    json.load(file)
                os.replace(temp_dir, checkpoint_dir)
                logger.info("Checkpoint saved and validated successfully at step %s", self.step)
            except Exception as exc:
                save_error = exc
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                logger.error("Failed to save checkpoint: %s", exc)
                import traceback
                logger.error(traceback.format_exc())

        if dist.is_initialized():
            save_status = torch.tensor(
                [0 if save_error is not None else 1],
                dtype=torch.int32,
                device=self.device,
            )
            dist.all_reduce(save_status, op=dist.ReduceOp.MIN)
            if save_status.item() == 0:
                raise RuntimeError("Checkpoint save or validation failed") from save_error
        elif save_error is not None:
            raise RuntimeError("Checkpoint save or validation failed") from save_error

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        step_in_accumulation = 0
        optimizer_step_started = None

        while self.step < self.config.num_steps:
            if step_in_accumulation == 0:
                optimizer_step_started = time.perf_counter()
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                step_in_accumulation = 0

                synchronize()
                step_time = time.perf_counter() - optimizer_step_started
                if self.step % self.config.gc_interval == 0:
                    empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += 1
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}',
                        'step_time': f'{step_time:.3f}s',
                    })
                    logger.info(
                        "PERF step=%d step_time=%.6f mbs=%d world_size=%d global_batch=%d",
                        self.step,
                        step_time,
                        self.config.batch_size,
                        self.config.world_size,
                        self.config.batch_size * self.config.world_size * self.gradient_accumulation_steps,
                    )
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=self.step)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    overrides = {
        "dataset_path": args.dataset_path,
        "wan22_pretrained_model_name_or_path": args.pretrained_model_path,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "dataset_init_worker": args.dataset_init_worker,
        "load_worker": args.load_worker,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_interval": args.save_interval,
        "cfg_prob": args.cfg_prob,
        "attn_mode": args.attn_mode,
    }
    for name, value in overrides.items():
        if value is not None:
            config[name] = value
    if config.get("dataset_path"):
        config.empty_emb_path = os.path.join(config.dataset_path, 'empty_emb.pt')
    if args.disable_wandb:
        config.enable_wandb = False

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    seed = os.environ.get("LINGBOT_SEED")
    if seed is not None:
        torch.manual_seed(int(seed) + rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override the configured training dataset root",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="Override the configured Wan2.2 pretrained model root",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the number of optimizer steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the per-rank micro batch size",
    )
    parser.add_argument(
        "--dataset-init-worker",
        type=int,
        default=None,
        help="Override the dataset metadata initialization worker count",
    )
    parser.add_argument(
        "--load-worker",
        type=int,
        default=None,
        help="Override the DataLoader worker count",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override the number of batches per optimizer step",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Override the checkpoint save interval",
    )
    parser.add_argument(
        "--cfg-prob",
        type=float,
        default=None,
        help="Override classifier-free text dropout probability",
    )
    parser.add_argument(
        "--attn-mode",
        choices=("flex", "torch", "flashattn"),
        default=None,
        help="Override the transformer attention implementation",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable WandB logging for this run",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
