import matplotlib.pyplot as plt
import re
import os

# Chemin vers votre fichier de log
log_path = r"checkpoints/fengwu_local_8gb/world_size1-fengwu_20years_batch4/iter.log"

def parse_logs(file_path):
    train_loss = []
    val_loss = []
    test_loss = []
    epochs = []

    # RegEx pour extraire les données
    # Exemple Train: Epoch [20/20][7325/7325] ... Possloss: 0.4862 (0.4970)
    train_pattern = re.compile(r"Epoch \[(\d+)/\d+\]\[\d+/\d+\].*Possloss: .* \(([\d\.]+)\)")
    # Exemple Val: Epoch [20](<val stats>)  Possloss: 0.4534 (0.4145)
    val_pattern = re.compile(r"Epoch \[(\d+)\]\(val stats\).*Possloss: .* \(([\d\.]+)\)")
    # Exemple Test: Epoch [20](<test stats>)  Possloss: 0.4838 (0.4145)
    test_pattern = re.compile(r"Epoch \[(\d+)\]\(test stats\).*Possloss: .* \(([\d\.]+)\)")

    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return None

    with open(file_path, 'r') as f:
        for line in f:
            # Extraction Train (on prend la dernière itération de chaque époque)
            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                # On met à jour ou on ajoute la perte de l'époque
                if len(epochs) < epoch:
                    epochs.append(epoch)
                    train_loss.append(loss)
                else:
                    train_loss[epoch-1] = loss

            # Extraction Val
            val_match = val_pattern.search(line)
            if val_match:
                val_loss.append(float(val_match.group(2)))

            # Extraction Test
            test_match = test_pattern.search(line)
            if test_match:
                test_loss.append(float(test_match.group(2)))

    return epochs, train_loss, val_loss, test_loss

# Analyse et traçage
data = parse_logs(log_path)
if data:
    epochs, t_loss, v_loss, te_loss = data

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, t_loss, label='Train Loss', marker='o', color='blue', linestyle='-')
    plt.plot(epochs[:len(v_loss)], v_loss, label='Validation Loss', marker='s', color='green', linestyle='--')
    
    if te_loss:
        plt.scatter(epochs[len(te_loss)-1], te_loss[-1], color='red', label='Final Test Loss', zorder=5, s=100)

    plt.title('FengWu-Lite Training Progress (20 Years Data)')
    plt.xlabel('Epoch')
    plt.ylabel('Possloss (RMSE)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Sauvegarde du graphique
    plt.savefig('training_results.png')
    print("Graphique sauvegardé sous 'training_results.png'")
    plt.show()
