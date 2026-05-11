#!/usr/bin/env python
import argparse
import os
import torch
import yaml

from utils.builder import ConfigBuilder
import utils.misc as utils
from utils.logger import get_logger

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger(
        "test",
        args.run_dir,
        utils.get_rank(),
        filename="test.log",
        resume=False
    )

    args.cfg_params["logger"] = logger

    if torch.cuda.is_available():
        device_id = args.local_rank
        torch.cuda.set_device(device_id)
        logger.info(f"GPU: {torch.cuda.get_device_name(device_id)}")
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("CUDA not available. Testing will run on CPU.")

    logger.info("Building config ...")
    builder = ConfigBuilder(**args.cfg_params)

    logger.info("Building test dataloader ...")
    test_dataloader = builder.get_dataloader(split="test")
    logger.info(f"Test dataloader build complete | steps: {len(test_dataloader)}")

    logger.info("Building model ...")
    model = builder.get_model()

    # Choisir le meilleur checkpoint ou le dernier
    model_checkpoint = os.path.join(args.run_dir, "checkpoint_best.pth")
    if not os.path.exists(model_checkpoint):
        model_checkpoint = os.path.join(args.run_dir, "checkpoint_latest.pth")

    if os.path.exists(model_checkpoint):
        logger.info(f"Loading checkpoint: {model_checkpoint}")
        model.load_checkpoint(model_checkpoint, resume=True)
    else:
        logger.warning(f"NO CHECKPOINT FOUND AT {args.run_dir}. Model is initialized randomly!")

    logger.info("Begin testing ...")
    
    # On limite le test à 50 étapes par défaut pour avoir un aperçu rapide, 
    # sinon on fait toute la boucle si l'utilisateur modifie le script
    MAX_TEST_STEPS = 50
    logger.info(f"Aperçu rapide : test limité aux {MAX_TEST_STEPS} premières étapes.")
    
    model_key = list(model.model.keys())[0]
    model.model[model_key].eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            if step >= MAX_TEST_STEPS:
                break
                
            loss = model.test_one_step(batch)
            metric_logger.update(**loss)
            
            if step % 10 == 0:
                print(f"[TEST RUNNING] Step {step}/{MAX_TEST_STEPS} | Loss: {list(loss.values())[0]:.4f}")

    logger.info('  '.join(
        [f'Test Results (Aperçu {MAX_TEST_STEPS} étapes):', "{meters}"]
    ).format(meters=str(metric_logger)))

def main(args):
    print("FengWu-Lite - Script de Test Rapide")
    args.rank = 0
    args.distributed = False
    args.local_rank = args.cuda

    desc = f"world_size{args.world_size:d}"
    if args.desc is not None and args.desc.strip() != "":
        desc += f"-{args.desc}"

    alg_dir = os.path.splitext(os.path.basename(args.cfg))[0]
    args.outdir = os.path.join(args.outdir, alg_dir)
    run_dir = os.path.join(args.outdir, desc)

    print(f"Loading config from {args.cfg} ...")
    with open(args.cfg, "r", encoding="utf-8") as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

    args.cfg_params = cfg_params
    args.run_dir = run_dir

    subprocess_fn(args)
    print("Testing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FengWu-Lite Model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="./checkpoints")
    parser.add_argument("--cfg", "-c", type=str, default=os.path.join("config", "fengwu_local_8gb.yaml"))
    parser.add_argument("--desc", type=str, default="fengwu_lite_64_37levels_ar")
    
    args = parser.parse_args()
    main(args)
