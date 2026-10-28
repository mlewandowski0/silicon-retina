import argparse
import json
import wandb
from datetime import datetime
import torch.optim as optim
import torch
import os
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity, schedule


from torch.utils.data import DataLoader
from tqdm import tqdm

from src.neural_network.networks.FIRE_network import FIRE_network
from src.neural_network.networks.SaliencyLoss import SaliencyLoss
from src.neural_network.dataloaders.SaliencyDatasetWrapper import SaliencyDatasetWrapper
from src.neural_network.dataloaders.reconstruction_utils import save_checkpoint  
from src.neural_network.dataloaders.EventDatasetFromUpsampledSaliency import EventDatasetFromUpsampledSaliency
from src.neural_network.utils.util import validate_occupancy_dataset

from torch.optim.lr_scheduler import StepLR


torch.backends.cudnn.benchmark = True

class StepLRWithMin(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, min_lr=1e-6):
        super().__init__(optimizer, step_size, gamma, last_epoch)
        self.min_lr = min_lr

    def get_lr(self):
        lrs = super().get_lr()
        return [max(lr, self.min_lr) for lr in lrs]


def train_epoch(model, loader, optimizer, device, epoch, config,
                saliency_loss, profiler):

    model.train()
    total = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
    use_amp = (device.type == "cuda") and config.get("use_amp", False)

    # Choose precision: "fp16" (with GradScaler) or "bf16" (no scaling needed).
    amp_dtype_str = config.get("amp_dtype", "fp16").lower()  # "fp16" or "bf16"
    amp_dtype = torch.float16 if amp_dtype_str == "fp16" else torch.bfloat16

    # GradScaler is only needed for fp16 on CUDA
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(
        device_type,  # 'cuda' or 'cpu'
        enabled=(use_amp and amp_dtype is torch.float16)  # scaling only for fp16 on CUDA
    )

    burn_in_steps = int(config.get("burn_in_steps", 0))
    print("burn in steps", burn_in_steps)
    max_shift = int(config.get("shift_loss_max_offset", 1))

    for batch_idx, batch in enumerate(pbar):
        # shapes:
        # frames: (B, L, 1, H, W)
        # grids:  (B, L, 5, H, W)
        # flows:  (B, L, 2, H, W)  (optional)

        if profiler is not None:
            profiler.step()

        frames = batch['frames'].to(device)
        grids = batch['grids'].to(device)
        flows = batch.get('optical_flows', None)
        if flows is not None:
            flows = flows.to(device)
        valid = batch['valid_mask'].to(device)

        B, L, _, H, W = frames.shape

        # reset recurrent state with the current spatial size
        model.reset(input_shape=(H, W), batch_size=B)

        loss = torch.zeros((), device=device)
        valid_steps = 0
        kl_term_sum = 0.0
        cc_term_sum = 0.0

        Igt_t = None
        Ihat_t = None


        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            # warm-up
            with torch.no_grad():
                bi = min(burn_in_steps, L)
                for t in range(bi):
                    if not valid[:, t].any(): 
                        continue
                    _ = model(grids[:, t])

        # main pass
        for t in range(bi, L):

            x_t  = grids[:, t]      # (B, C, H, W)
            Igt_t = frames[:, t]    # (B, 1, H, W)

            # forward under autocast
            with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                Ihat_t = model(x_t)

            if not valid[:, t].any():
                continue

            # loss in fp32 (outside autocast)
            step_loss, kl_loss, cc_loss = saliency_loss(Ihat_t, Igt_t)
            loss = loss + step_loss

            kl_term_sum += float(kl_loss.detach().cpu().item())
            cc_term_sum += float(cc_loss.detach().cpu().item())
            valid_steps += 1

        # Optional debug save
        if config.get("debug_save", False) and (batch_idx % config.get("debug_iters", 100) == 0):
            if (Igt_t is not None) and (Ihat_t is not None):
                os.makedirs(config["debug_folder"], exist_ok=True)
                with torch.no_grad():
                    fig, axes = plt.subplots(2, min(4, B), figsize=(16, 8))
                    if axes.ndim == 1:
                        axes = axes[None, :]  # ensure 2D for consistent indexing
                    for i in range(min(4, B)):
                        axes[0, i].imshow(Igt_t[i, 0].detach().cpu().numpy(), cmap='gray')
                        axes[0, i].set_title('GT')
                        axes[0, i].axis('off')
                        axes[1, i].imshow(Ihat_t[i, 0].detach().cpu().numpy(), cmap='gray')
                        axes[1, i].set_title('Pred')
                        axes[1, i].axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(config['debug_folder'], f'epoch_{epoch}_batch_{batch_idx}.png'))
                    plt.close()

        # normalize by number of valid time steps (prevents dependence on L)
        if valid_steps == 0:
            continue
        loss = loss / valid_steps

        # optimize
        optimizer.zero_grad(set_to_none=True)


        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += float(loss.detach().cpu().item())

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total / (batch_idx + 1):.4f}",
            'lr': optimizer.param_groups[0]['lr'],
            'kl_sum': f"{(kl_term_sum/valid_steps):.3f}",
            'cc_sum': f"{(cc_term_sum/valid_steps):.3f}",
        })

        # Log to wandb if enabled
        if config.get('use_wandb', False) and batch_idx % 10 == 0:
            wandb.log({
                'train_loss_step': float(loss.detach().cpu().item()),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'kl_loss_sum': kl_term_sum / valid_steps,
                'cc_term_sum': cc_term_sum / valid_steps,
                'epoch': epoch,
                'step': epoch * len(loader) + batch_idx
            })
    return total / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device, epoch, config, saliency_loss):
    model.eval()
    total = 0.0

    burn_in_steps = int(config.get("burn_in_steps", 0))
    max_shift = int(config.get("shift_loss_max_offset", 1))

    pbar = tqdm(loader, desc=f"Validation {epoch + 1}/{config['num_epochs']}")

    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(device)      # (B, L, 1, H, W)
        grids = batch['grids'].to(device)        # (B, L, 5, H, W)
        flows = batch.get('optical_flows', None)
        if flows is not None:
            flows = flows.to(device)
        valid = batch['valid_mask'].to(device)   # (B, L)

        B, L, _, H, W = frames.shape
        model.reset(input_shape=(H, W), batch_size=B)

        loss = 0.0
        valid_steps = 0

        bi = min(burn_in_steps, L)
        for t in range(bi):
            if not valid[:, t].any(): continue
            _ = model(grids[:, t])

        for t in range(bi, L):
            x_t = grids[:, t]
            Igt_t = frames[:, t]
            
            Ihat_t = model(x_t)

            if not valid[:, t].any():
                continue

            loss_sample, _, _ = saliency_loss(Ihat_t, Igt_t)
            loss += loss_sample

            Ihat_prev = Ihat_t
            Igt_prev = Igt_t
            valid_steps += 1

        if valid_steps == 0:
            continue
        loss = loss / valid_steps
        total += float(loss.detach().cpu().item())

        pbar.set_postfix({'val_loss': f"{total / (batch_idx + 1):.4f}"})

    return total / max(1, len(loader))



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to json config file")
    args = ap.parse_args()

    config_file = args.config
    with open(config_file, "r") as f:
        config = json.load(f)

    print("FIRENetwork Training on Saliency Dataset")
    print("================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    INP = config.get("num_bins", 5)
    config["num_bins"] = INP

    model = FIRE_network(in_ch=INP,
                         hid1=config["hid1"],
                         ks1=config["ks1"],
                         add_bn=config["add_bn"],
                         out_mode=config["out_mode"]).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    
    # Silicon Retina Dataset parameters
    use_upsampled_from_OF = config.get("use_upsampled_from_OF", False)
    config["use_upsampled_from_OF"] = use_upsampled_from_OF

    # verbose flag
    verbose = config.get("verbose", False)
    config["verbose"] = verbose

    event_representation_params = config.get("event_representation_params", {})
    event_representation_params["num_bins"] = config.get("num_bins", 5)
    config["num_bins"] = event_representation_params["num_bins"]
    

    force_recomputing = config.get("force_recomputing", False)
    config["force_recomputing"] = force_recomputing

    log_input = config.get("log_input", True)
    config["log_input"] = log_input

    # upsampled video the DVS reads: "optical_flow" (all original DVS runs) or "fixed_200fps" (the retina's input)
    dvs_input_video = config.get("dvs_input_video", "optical_flow")
    config["dvs_input_video"] = dvs_input_video

    skip_frames = config.get("skip_frames", False)
    config["skip_frames"] = skip_frames

    use_lpips = config.get('use_lpips', False)
    config["use_lpips"] = use_lpips

    # parameters for dataset checking
    check_dataset = config.get("check_dataset", True)
    min_grid_occupancy = config.get("min_grid_occupancy", 0.001)
    max_grid_occupancy = config.get("max_grid_occupancy", 0.6)
    min_per_pixel_occupancy = config.get("min_per_pixel_occupancy", 0.0)
    max_per_pixel_occupancy = config.get("max_per_pixel_occupancy", 1.0) 

    train_base = EventDatasetFromUpsampledSaliency(directory=os.path.join("src", "neural_network", "data", "saliency_dataset_ready", "train"),
                                   mode="event",
                                   prediction_video_name="outputs.npz",
                                   preprocessing_params=event_representation_params,
                                   log_scale=log_input,
                                   force_recomputing=force_recomputing,
                                   limit=config["train_limit"],
                                   verbose=verbose,
                                   input_video=dvs_input_video)

    train_dataset = SaliencyDatasetWrapper(train_base,
                                           sequence_length=config['sequence_length'],
                                           skip_frames=config.get('skip_frames', 1),
                                           augment=config["augment"])

    val_base = EventDatasetFromUpsampledSaliency(directory=os.path.join("src", "neural_network", "data", "saliency_dataset_ready", "val"),
                                   mode="event",
                                   prediction_video_name="outputs.npz",
                                   preprocessing_params=event_representation_params,
                                   log_scale=log_input,
                                   force_recomputing=force_recomputing,
                                   limit=config["val_limit"],
                                   verbose=verbose,
                                   input_video=dvs_input_video)

    val_dataset = SaliencyDatasetWrapper(val_base,
                                         sequence_length=config['sequence_length'],
                                         skip_frames=skip_frames,
                                         augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
        prefetch_factor=config['prefetch_factor'])

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False,
        prefetch_factor=config['prefetch_factor'])

    # profiling
    profile_on = config.get("profile_on", False)
    if profile_on:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=5, warmup=5, active=20, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                config.get("profile_dir", "./tb_prof")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )
        prof.__enter__()
    else:
        prof = None
        

    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Val dataset: {len(val_dataset)} sequences")

    # get result file
    result_file = config.get("result_file", "result.json")

    # check dataset if needed 
    if check_dataset:
        valid, grids_occupancy_train, per_pixel_occupany_train = validate_occupancy_dataset(train_base, 
                                                                                min_grid_occupancy=min_grid_occupancy, 
                                                                                max_grid_occupancy=max_grid_occupancy,
                                                                                min_per_pixel_occupancy=min_per_pixel_occupancy,
                                                                                max_per_pixel_occupancy=max_per_pixel_occupancy,
                                                                                result_file=result_file)


    # check dataset if needed 
    if check_dataset:
        valid, grids_occupancy_val, per_pixel_occupany_val = validate_occupancy_dataset(val_base, 
                                                                                min_grid_occupancy=min_grid_occupancy, 
                                                                                max_grid_occupancy=max_grid_occupancy,
                                                                                min_per_pixel_occupancy=min_per_pixel_occupancy,
                                                                                max_per_pixel_occupancy=max_per_pixel_occupancy,
                                                                                result_file=result_file)

        total = len(train_base) + len(val_base)
        grids_occupancy = (grids_occupancy_train * len(train_base) + grids_occupancy_val * len(val_base)) / total
        per_pixel_occupany = (per_pixel_occupany_train * len(train_base) + per_pixel_occupany_val * len(val_base)) / total
        print(f"Average occupancy over train and val datasets: Grids occupancy: {grids_occupancy*100:.4f} %, Per pixel occupancy: {per_pixel_occupany*100:.4f} %")
        
        config["grids_occupancy"] = grids_occupancy
        config["per_pixel_occupany"] = per_pixel_occupany
    
    
    use_wandb = config.get("use_wandb", False)
    config["use_wandb"] = use_wandb
    
    name_wandb = config.get("wandb_name", f"FIREnetwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if config.get('use_wandb', False):
        wandb.init(
            project="e2vid-saliency2",
            config=config,
            name=name_wandb
        )
        
    # print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Losses and optimizer
    loss_func = SaliencyLoss(eps=1e-4)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, config,
            saliency_loss=loss_func, profiler=prof)

        val_loss = validate(
            model, val_loader, device, epoch, config,
            saliency_loss=loss_func)

        print(f"\nTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if config.get('use_wandb', False):
            wandb.log({
                'train_loss_epoch': train_loss,
                'val_loss_epoch': val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch
            })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % config['save_every'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, config, is_best
            )

        scheduler.step()

    if prof is not None:
        prof.__exit__(None, None, None)
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
