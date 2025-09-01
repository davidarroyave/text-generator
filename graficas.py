import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

with open("models/history.pkl", "rb") as f:
    history = pickle.load(f)

print("Claves en historial:", history.keys())
print("Valores train_loss:", history.get("train_loss"))
print("Valores epoch:", history.get("epoch"))
print("Valores eval_loss:", history.get("eval_loss"))

# Cargar historial
history_path = "models/history.pkl"
with open(history_path, "rb") as f:
    history = pickle.load(f)

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Obtener listas del historial
epochs = history.get("epoch", [])
train_loss = history.get("train_loss", [])
eval_loss = history.get("eval_loss", [])

# Si epochs está vacío o no coincide, crear índices para plotear
if not epochs or len(epochs) != len(train_loss):
    epochs_train = list(range(1, len(train_loss) + 1))
else:
    epochs_train = epochs

if not eval_loss:
    epochs_eval = []
else:
    if not epochs or len(epochs) != len(eval_loss):
        epochs_eval = list(range(1, len(eval_loss) + 1))
    else:
        epochs_eval = epochs

# Plot curvas de pérdida con chequeo de longitud
plt.figure(figsize=(8, 5))
if train_loss:
    plt.plot(epochs_train, train_loss, marker="o", label="Train Loss")
if eval_loss:
    plt.plot(epochs_eval, eval_loss, marker="o", label="Eval Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "loss_curves.jpg"), dpi=300)
plt.close()
print(f"Guardado: {output_dir}/loss_curves.jpg")

# Calcular y guardar perplexity si hay datos
if train_loss:
    train_ppl = np.exp(train_loss[-1])
    with open(os.path.join(output_dir, "perplexity_report.txt"), "w") as f:
        f.write(f"Train perplexity: {train_ppl:.4f}\n")
    print(f"Guardado: {output_dir}/perplexity_report.txt")
else:
    print("No se pudo calcular perplexity: falta train_loss.")


