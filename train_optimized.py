#!/usr/bin/env python
import argparse
import os
import shutil
import torch
import yaml

from utils.builder import ConfigBuilder
import utils.misc as utils
from utils.logger import get_logger


def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger(
        "train",
        args.run_dir,
        utils.get_rank(),
        filename="iter.log",
        resume=args.resume
    )

    args.cfg_params["logger"] = logger

    if torch.cuda.is_available():
        device_id = args.local_rank
        torch.cuda.set_device(device_id)

        logger.info(f"GPU: {torch.cuda.get_device_name(device_id)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB"
        )

        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        logger.info("CUDA not available. Training will run on CPU.")

    logger.info("Building config ...")
    builder = ConfigBuilder(**args.cfg_params)

    logger.info("Building dataloaders ...")
    train_dataloader = builder.get_dataloader(split="train")
    logger.info(f"Train dataloader build complete | steps: {len(train_dataloader)}")

    val_dataloader = builder.get_dataloader(split="val")
    logger.info(f"Val dataloader build complete   | steps: {len(val_dataloader)}")

    test_dataloader = builder.get_dataloader(split="test")
    logger.info(f"Test dataloader build complete  | steps: {len(test_dataloader)}")

    
    model_params = args.cfg_params["model"]["params"]

    extra_params = model_params.get("extra_params", {})
    replay_cfg = extra_params.get("replay_buff", None)

    if replay_cfg is not None:
        logger.info("Replay Buffer: ENABLED")
        logger.info(f"Replay config: {replay_cfg}")
    else:
        logger.info("Replay Buffer: DISABLED")

    steps_per_epoch = model_params.get("train_steps", len(train_dataloader))

    if "lr_scheduler" in model_params:
        lr_scheduler_params = model_params["lr_scheduler"]

        for model_key in lr_scheduler_params:
            if lr_scheduler_params[model_key].get("by_step", False):
                for param_key in lr_scheduler_params[model_key]:
                    if "epochs" in param_key:
                        lr_scheduler_params[model_key][param_key] *= steps_per_epoch

    logger.info("Building model ...")
    model = builder.get_model()

    model_checkpoint = os.path.join(args.run_dir, "checkpoint_latest.pth")

    if args.resume:
        if os.path.exists(model_checkpoint):
            logger.info(f"Resuming from checkpoint: {model_checkpoint}")
            model.load_checkpoint(model_checkpoint, resume=True)
        else:
            logger.warning(f"Resume requested but checkpoint not found: {model_checkpoint}")

    for key in model.model:
        params = [p for p in model.model[key].parameters() if p.requires_grad]
        cnt_params = sum(p.numel() for p in params)
        logger.info(f"Trainable params {key}: {cnt_params:,}")

    logger.info("Begin training ...")
    logger.info("Recommended: AMP=True, use_checkpoint=True, batch_size=1")

    model.trainer(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        builder.get_max_epoch(),
        checkpoint_savedir=args.run_dir,
        resume=args.resume,
        patience=builder.trainer_params.get('patience', 100)
    )


def main(args):
    print("FengWu-Lite Training - RTX A1000 / Colab / Local")
    print(f"Config: {args.cfg}")

    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.distributed = False
        args.local_rank = args.cuda

        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)

    desc = f"world_size{args.world_size:d}"

    if args.desc is not None and args.desc.strip() != "":
        desc += f"-{args.desc}"

    alg_dir = os.path.splitext(os.path.basename(args.cfg))[0]
    args.outdir = os.path.join(args.outdir, alg_dir)
    run_dir = os.path.join(args.outdir, desc)

    args.relative_checkpoint_dir = os.path.join(alg_dir, desc)

    print(f"Output dir: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)

    train_config_file = os.path.join(run_dir, "training_options.yaml")

    if (not args.resume) or args.resume_from_config or (not os.path.exists(train_config_file)):
        print("Loading config from YAML file...")
        with open(args.cfg, "r", encoding="utf-8") as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
    else:
        print("Loading config from resume file...")
        with open(train_config_file, "r", encoding="utf-8") as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

        arg_keys = set(vars(args).keys())
        for key in list(cfg_params.keys()):
            if key in arg_keys:
                del cfg_params[key]

    dataset_vnames = cfg_params["dataset"]["train"].get("vnames", None)

    if dataset_vnames is not None:
        constants_len = len(dataset_vnames.get("constants", []))
    else:
        constants_len = 0

    cfg_params["model"]["params"]["constants_len"] = constants_len

    if args.rank == 0:
        config_backup_file = os.path.join(run_dir, "used_config.yaml")
        shutil.copyfile(args.cfg, config_backup_file)

        with open(train_config_file, "w", encoding="utf-8") as f:
            yaml.dump(vars(args), f, indent=2, sort_keys=False)
            yaml.dump(cfg_params, f, indent=2, sort_keys=False)

    args.cfg_params = cfg_params
    args.run_dir = run_dir

    print("Launching training...")
    subprocess_fn(args)
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FengWu-Lite training optimized for RTX A1000 8GB"
    )

    parser.add_argument("--tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_model_parallel_size", type=int, default=1)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from_config", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--per_cpus", type=int, default=1)

    parser.add_argument(
        "--init_method",
        type=str,
        default="tcp://127.0.0.1:23456"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./checkpoints"
    )

    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        default=os.path.join("config", "fengwu_local_8gb.yaml"),
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--desc",
        type=str,
        default="fengwu_lite_64_37levels_ar",
        help="Experiment name"
    )

    args = parser.parse_args()
    main(args)