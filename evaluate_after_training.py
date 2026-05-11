#!/usr/bin/env python
import os
import argparse
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt

from utils.builder import ConfigBuilder
import utils.misc as utils
from utils.logger import get_logger


VAR_NAMES = ["u10", "v10", "t2m", "msl"]


def compute_metrics(pred, target):
    """
    pred, target : [B, C, H, W]
    On évalue les 4 premiers canaux : u10, v10, t2m, msl.
    """
    pred = pred[:, :4, :, :].float()
    target = target[:, :4, :, :].float()

    diff = pred - target

    rmse = torch.sqrt(torch.mean(diff ** 2, dim=(0, 2, 3)))
    mae = torch.mean(torch.abs(diff), dim=(0, 2, 3))

    pearson_list = []

    for c in range(4):
        p = pred[:, c, :, :].reshape(-1)
        t = target[:, c, :, :].reshape(-1)

        p = p - p.mean()
        t = t - t.mean()

        corr = torch.sum(p * t) / (
            torch.sqrt(torch.sum(p ** 2)) * torch.sqrt(torch.sum(t ** 2)) + 1e-8
        )

        pearson_list.append(corr)

    pearson = torch.stack(pearson_list)

    return rmse, mae, pearson


def main():
    parser = argparse.ArgumentParser(description="Evaluation finale FengWu-Lite")

    parser.add_argument("--cfg", type=str, default="config/fengwu_local_8gb.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="evaluation_results")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading config from {args.cfg} ...")

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg_params = yaml.load(f, Loader=yaml.FullLoader)

    # Correction Windows : éviter blocage DataLoader
    if "dataloader" not in cfg_params:
        cfg_params["dataloader"] = {}
    cfg_params["dataloader"]["num_workers"] = 0
    cfg_params["dataloader"]["pin_memory"] = False

    cfg_params["seed"] = 0
    cfg_params["world_size"] = 1
    cfg_params["rank"] = 0
    cfg_params["local_rank"] = 0

    logger = get_logger(
        "evaluate_final",
        args.outdir,
        0,
        filename="evaluate_final.log",
        resume=False
    )

    cfg_params["logger"] = logger

    utils.setup_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("Building test dataloader ...", flush=True)
    builder = ConfigBuilder(**cfg_params)
    test_loader = builder.get_dataloader(split="test")

    print("Building model ...", flush=True)
    model = builder.get_model()

    if args.checkpoint is None:
        run_dir = os.path.join(
            "./checkpoints",
            "fengwu_local_8gb",
            "world_size1-fengwu_20years_batch4"
        )

        checkpoint_path = os.path.join(run_dir, "checkpoint_best.pth")

        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(run_dir, "checkpoint_latest.pth")
    else:
        checkpoint_path = args.checkpoint

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    model.load_checkpoint(checkpoint_path, resume=True)

    model_key = list(model.model.keys())[0]
    net = model.model[model_key]
    net.to(device)
    net.eval()

    print("\nCheckpoint chargé correctement.", flush=True)
    print(f"\nEvaluation on {args.batches} batches...", flush=True)
    
    test_iter = iter(test_loader)
    
    rows = []

    with torch.no_grad():
        for batch_idx in range(args.batches):
            try:
                batch = next(test_iter)
            except StopIteration:
                print(f"\nFin du dataset atteinte prématurément à l'index {batch_idx}.", flush=True)
                break
            
            inp, target_seq = model.data_preprocess(batch)

            inp = inp.to(device).float()
            target_seq = target_seq.to(device).float()

            if target_seq.dim() != 5:
                raise ValueError(f"target_seq doit être [B,T,C,H,W], reçu : {target_seq.shape}")

            B, ar_steps, C, H, W = target_seq.shape
            current_input = inp

            for k in range(ar_steps):
                target = target_seq[:, k, :, :, :]

                output = net(current_input)

                if output.shape[1] == 2 * C:
                    pred_mean, log_var = torch.chunk(output, 2, dim=1)
                elif output.shape[1] == C:
                    pred_mean = output
                else:
                    raise ValueError(
                        f"Sortie modèle inattendue : {output.shape}, attendu C={C} ou 2C={2*C}"
                    )

                rmse, mae, pearson = compute_metrics(pred_mean, target)

                horizon_h = (k + 1) * 6

                for i, var in enumerate(VAR_NAMES):
                    rows.append({
                        "batch": batch_idx + 1,
                        "horizon_h": horizon_h,
                        "variable": var,
                        "rmse": rmse[i].item(),
                        "mae": mae[i].item(),
                        "pearson": pearson[i].item()
                    })

                current_input = torch.cat(
                    [current_input[:, C:, :, :], pred_mean],
                    dim=1
                )

            print(f"Batch {batch_idx + 1}/{args.batches} évalué.")

    print("\nCalcul des moyennes finales...", flush=True)
    if len(rows) == 0:
        raise RuntimeError("Aucun batch évalué. Vérifie ton dataloader ou le paramètre --batches.")

    print(f"Extraction de {len(rows)} lignes de données...", flush=True)
    
    # --- Sauvegarde CSV Détaillé (sans Pandas) ---
    import csv
    detailed_csv = os.path.join(args.outdir, "metrics_by_batch.csv")
    print(f"Sauvegarde du CSV détaillé dans {detailed_csv}...", flush=True)
    
    keys = rows[0].keys()
    with open(detailed_csv, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(rows)

    # --- Calcul du résumé (sans Pandas) ---
    print("Calcul du résumé par horizon...", flush=True)
    summary_data = {} # (horizon, var) -> [rmse_sum, mae_sum, p_sum, count]
    
    for r in rows:
        key = (r["horizon_h"], r["variable"])
        if key not in summary_data:
            summary_data[key] = [0.0, 0.0, 0.0, 0]
        summary_data[key][0] += r["rmse"]
        summary_data[key][1] += r["mae"]
        summary_data[key][2] += r["pearson"]
        summary_data[key][3] += 1
        
    summary_list = []
    for (h, v), vals in summary_data.items():
        summary_list.append({
            "horizon_h": h,
            "variable": v,
            "rmse": vals[0] / vals[3],
            "mae": vals[1] / vals[3],
            "pearson": vals[2] / vals[3]
        })
    
    # Trier par horizon et variable
    summary_list.sort(key=lambda x: (x["horizon_h"], x["variable"]))

    # --- Sauvegarde CSV Résumé ---
    summary_csv = os.path.join(args.outdir, "metrics_summary.csv")
    print(f"Sauvegarde du CSV résumé dans {summary_csv}...", flush=True)
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=summary_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(summary_list)

    print("\n==============================")
    print("RÉSUMÉ GLOBAL")
    print("==============================")
    print(f"{'Horizon':<8} | {'Var':<5} | {'RMSE':<8} | {'MAE':<8} | {'Pearson':<8}")
    for s in summary_list:
        print(f"{s['horizon_h']:<8} | {s['variable']:<5} | {s['rmse']:<8.4f} | {s['mae']:<8.4f} | {s['pearson']:<8.4f}")

    print(f"\nCSV détaillé sauvegardé : {detailed_csv}")
    print(f"CSV résumé sauvegardé   : {summary_csv}")

    for metric in ["rmse", "mae", "pearson"]:
        plt.figure(figsize=(8, 5))

        for var in VAR_NAMES:
            # Filtrer manuellement dans la liste
            sub_horizons = [s["horizon_h"] for s in summary_list if s["variable"] == var]
            sub_values = [s[metric] for s in summary_list if s["variable"] == var]
            
            plt.plot(sub_horizons, sub_values, marker="o", label=var)

        plt.xlabel("Horizon de prévision (heures)")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} selon l'horizon de prévision")
        plt.legend()
        plt.grid(True)

        fig_path = os.path.join(args.outdir, f"{metric}_by_horizon.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Graphique sauvegardé : {fig_path}", flush=True)

    # --- VISUALISATION MÉTÉO (CARTES ET MARRAKECH) ---
    print("\n===== GÉNÉRATION DES VISUALISATIONS (CARTES & MARRAKECH) =====", flush=True)
    
    # Coordonnées Marrakech approx (Grille 64x64)
    marrakech_lat_idx = 20
    marrakech_lon_idx = 30

    print("Démarrage de la génération des cartes...", flush=True)

    with torch.no_grad():
        # On recharge un batch frais pour la démo visuelle
        test_loader_vis = builder.get_dataloader(split="test")
        batch = next(iter(test_loader_vis))
        inp, target_seq = model.data_preprocess(batch)
        
        inp = inp.to(device).float()
        target_seq = target_seq.to(device).float()
        
        current_input = inp
        predictions = []
        
        for k in range(target_seq.shape[1]):
            output = net(current_input)
            if output.shape[1] == 2 * target_seq.shape[2]:
                pred_mean, _ = torch.chunk(output, 2, dim=1)
            else:
                pred_mean = output
            
            predictions.append(pred_mean)
            C = target_seq.shape[2]
            current_input = torch.cat([current_input[:, C:, :, :], pred_mean], dim=1)
        
        predictions = torch.stack(predictions, dim=1) 

        # 1. Cartes (Horizon +24h)
        h_idx = 3 
        for i, var in enumerate(VAR_NAMES):
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(target_seq[0, h_idx, i].cpu(), cmap='RdYlBu_r')
            plt.title(f"Réel {var} (+24h)")
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(predictions[0, h_idx, i].cpu(), cmap='RdYlBu_r')
            plt.title(f"Prédit {var} (+24h)")
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            diff = (predictions[0, h_idx, i] - target_seq[0, h_idx, i]).cpu()
            plt.imshow(diff, cmap='bwr')
            plt.title("Erreur (P-R)")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"map_comparison_{var}.png"))
            plt.close()

        # 2. Courbes Marrakech
        for i, var in enumerate(VAR_NAMES):
            plt.figure(figsize=(10, 5))
            
            real_series = target_seq[0, :, i, marrakech_lat_idx, marrakech_lon_idx].cpu()
            pred_series = predictions[0, :, i, marrakech_lat_idx, marrakech_lon_idx].cpu()
            horizons = [(k+1)*6 for k in range(len(real_series))]
            
            plt.plot(horizons, real_series, 'g-o', label="Vérité Terrain")
            plt.plot(horizons, pred_series, 'r--x', label="Prédiction FengWu")
            
            plt.title(f"Série Temporelle {var} - Point Marrakech")
            plt.xlabel("Heures de prévision")
            plt.ylabel("Valeur normalisée")
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(args.outdir, f"marrakech_curve_{var}.png"))
            plt.close()

    print(f"\n✅ Cartes et courbes Marrakech générées dans : {args.outdir}", flush=True)
    print("\n✅ Évaluation terminée avec succès.", flush=True)


if __name__ == "__main__":
    main()