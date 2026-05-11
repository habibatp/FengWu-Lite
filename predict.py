#!/usr/bin/env python
import argparse
import os
import torch
import yaml
import numpy as np

from utils.builder import ConfigBuilder
import utils.misc as utils
from utils.logger import get_logger


def load_and_normalize(file1, file2):
    """
    Charge deux fichiers .npy : T-6h et T.
    Chaque fichier doit avoir la forme [C, H, W].
    """
    data1 = np.load(file1).astype(np.float32)
    data2 = np.load(file2).astype(np.float32)

    if data1.shape != data2.shape:
        raise ValueError(f"Les deux fichiers n'ont pas la même shape : {data1.shape} vs {data2.shape}")

    # [T, C, H, W]
    seq = np.stack([data1, data2], axis=0)

    # Normalisation locale provisoire
    # Attention : idéalement, il faut utiliser les mean/std du training dataset
    mean = seq.mean(axis=(0, 2, 3), keepdims=True)  # [1, C, 1, 1]
    std = seq.std(axis=(0, 2, 3), keepdims=True)    # [1, C, 1, 1]
    std[std == 0] = 1.0

    seq_norm = (seq - mean) / std

    # inp : [B, T, C, H, W]
    inp = torch.from_numpy(seq_norm).unsqueeze(0)

    # mean/std : [1, C, 1, 1]
    mean = torch.from_numpy(mean).squeeze(0)
    std = torch.from_numpy(std).squeeze(0)

    return inp, mean, std


def main():
    parser = argparse.ArgumentParser(description="Inférence FengWu-Lite sur plusieurs pas de temps")

    parser.add_argument("--file1", type=str, required=True, help="Fichier .npy de l'instant T-6h")
    parser.add_argument("--file2", type=str, required=True, help="Fichier .npy de l'instant T")
    parser.add_argument("--steps", type=int, default=12, help="12 étapes = 3 jours avec pas de 6h")
    parser.add_argument("--outdir", type=str, default="./predictions")
    parser.add_argument("--cfg", type=str, default=os.path.join("config", "fengwu_local_8gb.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading config from {args.cfg} ...")
    with open(args.cfg, "r", encoding="utf-8") as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

    cfg_params["seed"] = 0
    cfg_params["world_size"] = 1
    cfg_params["rank"] = 0
    cfg_params["local_rank"] = 0

    utils.setup_seed(0)

    logger = get_logger("predict", args.outdir, 0, filename="predict.log", resume=False)
    cfg_params["logger"] = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building model ...")
    builder = ConfigBuilder(**cfg_params)
    model = builder.get_model()

    if args.checkpoint is not None:
        model_checkpoint = args.checkpoint
    else:
        run_dir = os.path.join(
            "./checkpoints",
            "fengwu_local_8gb",
            "world_size1-fengwu_lite_64_37levels_ar"
        )
        model_checkpoint = os.path.join(run_dir, "checkpoint_best.pth")

        if not os.path.exists(model_checkpoint):
            model_checkpoint = os.path.join(run_dir, "checkpoint_latest.pth")

    if os.path.exists(model_checkpoint):
        print(f"Loading checkpoint: {model_checkpoint}")
        model.load_checkpoint(model_checkpoint, resume=True)
    else:
        raise FileNotFoundError(f"Checkpoint non trouvé : {model_checkpoint}")

    model_key = list(model.model.keys())[0]
    net = model.model[model_key]
    net.eval()
    net.to(device)

    print("Chargement et normalisation des données...")
    inp, mean, std = load_and_normalize(args.file1, args.file2)

    inp = inp.to(device).float()
    mean = mean.to(device).float()  # [C, 1, 1]
    std = std.to(device).float()    # [C, 1, 1]

    B, T, C, H, W = inp.shape

    if T != 2:
        raise ValueError(f"Le modèle attend 2 pas temporels, mais T={T}")

    # [B, T*C, H, W]
    current_input = inp.reshape(B, T * C, H, W)

    print(f"\n--- Début de l'inférence : {args.steps} étapes = {(args.steps * 6) / 24:.1f} jours ---")

    with torch.no_grad():
        for k in range(args.steps):
            predict = net(current_input)

            # Le modèle sort souvent [mean, log_var]
            pred_mean, log_var = torch.chunk(predict, 2, dim=1)

            if pred_mean.shape[1] != C:
                raise ValueError(
                    f"Nombre de canaux prédit différent : pred={pred_mean.shape[1]}, attendu={C}"
                )

            # Dé-normalisation : [B, C, H, W]
            pred_unnorm = pred_mean * std.unsqueeze(0) + mean.unsqueeze(0)

            out_filename = os.path.join(
                args.outdir,
                f"prediction_step_{k+1:02d}_Tplus{(k+1)*6}h.npy"
            )

            np.save(out_filename, pred_unnorm.squeeze(0).cpu().numpy())

            print(
                f"Étape {k+1}/{args.steps} | +{(k+1)*6}h sauvegardée : {out_filename}"
            )

            # Autorégressif :
            # on enlève l'ancien T-6h et on ajoute la prédiction normalisée
            current_input = torch.cat(
                [current_input[:, C:, :, :], pred_mean],
                dim=1
            )

    print("\n✅ Prédictions terminées avec succès.")


if __name__ == "__main__":
    main()