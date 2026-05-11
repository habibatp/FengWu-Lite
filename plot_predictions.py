import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_comparison(pred_file, truth_file, step_name, channel_to_plot=0):
    if not os.path.exists(pred_file):
        print(f"Fichier de prédiction introuvable : {pred_file}")
        sys.exit(1)
    if not os.path.exists(truth_file):
        print(f"Fichier de vérité terrain introuvable : {truth_file}")
        sys.exit(1)

    # Chargement des données (shape: [189, 64, 64])
    pred_data = np.load(pred_file)
    truth_data = np.load(truth_file)

    # Extraction d'un canal spécifique
    pred_img = pred_data[channel_to_plot]
    truth_img = truth_data[channel_to_plot]

    # Calcul de l'erreur absolue
    error_img = np.abs(pred_img - truth_img)

    # Création du graphique
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Min/Max pour avoir la même échelle de couleurs
    vmin = min(truth_img.min(), pred_img.min())
    vmax = max(truth_img.max(), pred_img.max())

    im0 = axs[0].imshow(truth_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Vérité terrain ({step_name})")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(pred_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Prédiction FengWu ({step_name})")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(error_img, cmap='Reds')
    axs[2].set_title(f"Erreur Absolue")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Comparaison Prédiction vs Vérité - {step_name} (Canal {channel_to_plot})", fontsize=16)
    plt.tight_layout()
    
    out_img = f"comparaison_{step_name.replace('+', '').replace('h', '')}.png"
    plt.savefig(out_img, dpi=150)
    print(f"Graphique sauvegarde sous '{out_img}'")
    
    # Affichage interactif si possible
    plt.show()

if __name__ == "__main__":
    # Étape 1 : +6h
    # Les entrées étaient 2016_01_000.npy et 2016_01_001.npy. La première prédiction (+6h) correspond donc au fichier suivant : 2016_01_002.npy
    
    pred_step_1 = "predictions/prediction_step_01_Tplus6h.npy"
    truth_step_1 = "C:/Users/user/Desktop/Graphcast_Project/ERA5_np_float32_2016_2026/2016_01_002.npy"
    
    print("Génération de la comparaison pour +6h...")
    plot_comparison(pred_step_1, truth_step_1, step_name="+6h", channel_to_plot=0)
    
    # Étape 2 : +12h (correspond à 2016_01_003.npy)
    pred_step_2 = "predictions/prediction_step_02_Tplus12h.npy"
    truth_step_2 = "C:/Users/user/Desktop/Graphcast_Project/ERA5_np_float32_2016_2026/2016_01_003.npy"
    
    if os.path.exists(pred_step_2) and os.path.exists(truth_step_2):
        print("\nGénération de la comparaison pour +12h...")
        plot_comparison(pred_step_2, truth_step_2, step_name="+12h", channel_to_plot=0)
