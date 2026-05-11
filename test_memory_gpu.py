#!/usr/bin/env python
"""
Script de test d'utilisation mémoire GPU pour LGUnet_all
Compatible Windows (sans emojis)
"""

import torch
import sys
import os

# Ajouter le repo au path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from networks.LGUnet_all import LGUnet_all


def test_memory_usage():

    print("=" * 80)
    print("TEST MEMOIRE GPU - LGUnet_all (RTX A1000 8GB)")
    print("=" * 80)

    # ======================
    # GPU CHECK
    # ======================
    if not torch.cuda.is_available():
        print("CUDA non disponible!")
        return False

    device = torch.device("cuda:0")

    print(f"GPU detecte: {torch.cuda.get_device_name(0)}")
    print(f"Memoire totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # ======================
    # CONFIG MODELE (64x64)
    # ======================
    config = {
        "img_size": [64, 64],
        "patch_size": [2, 2],
        "stride": [2, 2],

        "inchans_list": [4, 37, 37, 37, 37, 37, 4, 37, 37, 37, 37, 37],
        "outchans_list": [8, 74, 74, 74, 74, 74],

        "in_chans": 378,
        "out_chans": 378,

        "enc_dim": 32,
        "embed_dim": 256,

        "window_size": [4, 8],

        "enc_depths": [1, 1, 1],
        "enc_heads": [2, 4, 4],

        "lg_depths": [1, 1],
        "lg_heads": [4, 4],

        "Weather_T": 1,
        "drop_path": 0.0,
        "use_checkpoint": True,
        "inp_length": 2,
        "use_mlp": False,
    }

    print("Configuration du modele:")
    print(f"Resolution: {config['img_size']}")
    print(f"Canaux entree: {config['in_chans']}")
    print(f"embed_dim: {config['embed_dim']}")
    print(f"enc_dim: {config['enc_dim']}")
    print()

    # ======================
    # CREATION MODELE
    # ======================
    print("Creation du modele...")

    try:
        model = LGUnet_all(**config).to(device)
        print("Modele cree avec succes")
    except Exception as e:
        print("Erreur creation modele:", e)
        return False

    # ======================
    # PARAMETRES
    # ======================
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametres: {total_params:,}")
    print()

    # ======================
    # TEST FORWARD
    # ======================
    print("Test forward pass...")

    torch.cuda.reset_peak_memory_stats()

    try:
        # input correct FengWu-Lite
        input_tensor = torch.randn(1, 378, 64, 64, device=device)

        print(f"Input shape: {input_tensor.shape}")

        with torch.no_grad():
            output = model(input_tensor)

        print(f"Output shape: {output.shape}")
        print("Forward OK")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OUT OF MEMORY:", e)
        else:
            print("Erreur:", e)
        return False

    # ======================
    # MEMOIRE GPU
    # ======================
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    print()
    print("Utilisation GPU:")
    print(f"Peak memory: {peak_memory:.2f} GB")

    if peak_memory > 7.5:
        print("WARNING: proche limite 8GB")
    else:
        print(f"OK - {8.0 - peak_memory:.2f} GB libres")

    print()
    print("=" * 80)
    print("TEST PASSE - MODELE COMPATIBLE GPU")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_memory_usage()
    sys.exit(0 if success else 1)