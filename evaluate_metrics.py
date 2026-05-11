#!/usr/bin/env python
import argparse
import os
import torch
import yaml
import numpy as np

from utils.builder import ConfigBuilder
import utils.misc as utils
from utils.logger import get_logger

def calculate_metrics(pred, target):
    """
    Calcule RMSE, MAE, et Pearson Correlation pour les 4 premiers canaux.
    pred et target shape: [B, C, H, W]
    """
    # RMSE
    mse = torch.mean((pred - target) ** 2, dim=(0, 2, 3))
    rmse = torch.sqrt(mse)
    
    # MAE
    mae = torch.mean(torch.abs(pred - target), dim=(0, 2, 3))
    
    # Pearson Correlation
    pred_mean = torch.mean(pred, dim=(0, 2, 3), keepdim=True)
    target_mean = torch.mean(target, dim=(0, 2, 3), keepdim=True)
    
    pred_var = pred - pred_mean
    target_var = target - target_mean
    
    cov = torch.mean(pred_var * target_var, dim=(0, 2, 3))
    pred_std = torch.sqrt(torch.mean(pred_var ** 2, dim=(0, 2, 3)))
    target_std = torch.sqrt(torch.mean(target_var ** 2, dim=(0, 2, 3)))
    
    # Éviter la division par zéro
    pearson = cov / (pred_std * target_std + 1e-8)
    
    # ACC (simplifié ici car on n'a pas la climatologie globale, on utilise la moyenne spatiale)
    acc = pearson.clone()
    
    return rmse, mae, pearson, acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate Metrics for FengWu-Lite")
    parser.add_argument("--cfg", "-c", type=str, default=os.path.join("config", "fengwu_local_8gb.yaml"))
    parser.add_argument("--batches", type=int, default=5, help="Nombre de lots à évaluer pour l'aperçu")
    args = parser.parse_args()

    print(f"Loading config from {args.cfg} ...")
    with open(args.cfg, "r", encoding="utf-8") as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

    # Paramètres par défaut
    cfg_params['seed'] = 0
    cfg_params['world_size'] = 1
    cfg_params['rank'] = 0
    cfg_params['local_rank'] = 0
    
    logger = get_logger("evaluate", "./checkpoints", 0, filename="evaluate.log", resume=False)
    cfg_params['logger'] = logger
    
    utils.setup_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    builder = ConfigBuilder(**cfg_params)
    print("Building test dataloader ...")
    test_dataloader = builder.get_dataloader(split="test")

    print("Building model ...")
    model = builder.get_model()

    run_dir = os.path.join("./checkpoints", "fengwu_local_8gb", "world_size1-fengwu_lite_64_37levels_ar")
    model_checkpoint = os.path.join(run_dir, "checkpoint_best.pth")

    if os.path.exists(model_checkpoint):
        print(f"Loading checkpoint: {model_checkpoint}")
        model.load_checkpoint(model_checkpoint, resume=True)
    else:
        print(f"WARNING: No checkpoint found at {model_checkpoint}")

    model_key = list(model.model.keys())[0]
    model.model[model_key].eval()
    model.model[model_key].to(device)

    vars_names = ['u10', 'v10', 't2m', 'msl']
    
    total_rmse = torch.zeros(4).to(device)
    total_mae = torch.zeros(4).to(device)
    total_pearson = torch.zeros(4).to(device)
    total_acc = torch.zeros(4).to(device)
    count = 0

    print(f"\n--- Début de l'évaluation sur {args.batches} lots ---")
    
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            if step >= args.batches:
                break
                
            inp, target_seq = model.data_preprocess(batch)
            
            ar_steps = target_seq.shape[1]
            C = target_seq.shape[2]
            current_input = inp
            
            # Prédiction autorégressive
            step_rmse, step_mae, step_pearson, step_acc = 0, 0, 0, 0
            
            for k in range(ar_steps):
                target = target_seq[:, k]
                
                predict = model.model[model_key](current_input)
                mean, log_var = torch.chunk(predict, 2, dim=1)
                
                # Récupérer les 4 premiers canaux (les variables de surface)
                pred_surface = mean[:, :4, :, :]
                target_surface = target[:, :4, :, :]
                
                rmse, mae, pearson, acc = calculate_metrics(pred_surface, target_surface)
                
                step_rmse += rmse
                step_mae += mae
                step_pearson += pearson
                step_acc += acc
                
                current_input = torch.cat([current_input[:, C:], mean], dim=1)
            
            # Moyenne sur les étapes autorégressives
            total_rmse += step_rmse / ar_steps
            total_mae += step_mae / ar_steps
            total_pearson += step_pearson / ar_steps
            total_acc += step_acc / ar_steps
            count += 1
            
            print(f"Batch {step+1}/{args.batches} évalué.")

    avg_rmse = (total_rmse / count).cpu().numpy()
    avg_mae = (total_mae / count).cpu().numpy()
    avg_pearson = (total_pearson / count).cpu().numpy()
    avg_acc = (total_acc / count).cpu().numpy()

    print("\n" + "="*60)
    print("RÉSULTATS DE L'ÉVALUATION (Données Normalisées)")
    print("ATTENTION : Ces résultats utilisent le checkpoint actuel (corrompu).")
    print("="*60)
    print(f"{'Variable':<10} | {'RMSE':<10} | {'MAE':<10} | {'Pearson':<10} | {'ACC':<10}")
    print("-" * 60)
    for i, var in enumerate(vars_names):
        print(f"{var:<10} | {avg_rmse[i]:<10.4f} | {avg_mae[i]:<10.4f} | {avg_pearson[i]:<10.4f} | {avg_acc[i]:<10.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
