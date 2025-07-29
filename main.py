import json
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

# === Étape 1 : Chargement des données depuis un fichier JSON ===
def load_data(file_path):
    print(f"[ÉTAPE 1] Chargement des données depuis : {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    images = [item['image'] for item in data]
    labels = [item['label'] for item in data]
    print(f"[OK] {len(images)} images chargées avec succès.")
    return images, labels

# === Étape 1 (suite) : Normalisation des pixels (entre 0 et 1) ===
def normalize_images(images):
    print("[ÉTAPE 1] Normalisation des pixels entre 0 et 1...")
    normalized = [[pixel / 255.0 for pixel in image] for image in images]
    print("[OK] Normalisation terminée.")
    return normalized

# === Étape 2 : Initialisation des poids et biais ===
def initialize_parameters(input_size, hidden_size, output_size):
    print(f"[ÉTAPE 2] Initialisation des poids et biais du réseau...")
    np.random.seed(0)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    print(f"[OK] Paramètres initialisés : W1({W1.shape}), b1({b1.shape}), W2({W2.shape}), b2({b2.shape})")
    return W1, b1, W2, b2

# === Fonction d’activation ReLU ===
def relu(z):
    return np.maximum(0, z)

# === Fonction softmax ===
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # stabilité numérique
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# === Fonction de coût : entropie croisée ===
def cross_entropy_loss(probs, label):
    return -np.log(probs[0, label] + 1e-15)  # éviter log(0)

# === Étape 3 : Propagation avant (calcul de la prédiction) ===
def forward_propagation(image, W1, b1, W2, b2):
    Z1 = np.dot(image, W1) + b1         # Somme pondérée entrée → cachée
    A1 = relu(Z1)                       # Activation ReLU
    Z2 = np.dot(A1, W2) + b2            # Somme pondérée cachée → sortie
    return Z1, A1, Z2

# === Étape 4 : Entraînement (correction simple, avec félicitations/punitions) ===
def train(images, labels, W1, b1, W2, b2, learning_rate=0.01, epochs=10):
    print("[ÉTAPE 4] Début de l'entraînement avec la règle de correction simplifiée...")
    epoch_accuracies = []
    epoch_losses = []

    for epoch in range(epochs):
        print(f"\n[ÉPOQUE {epoch + 1}/{epochs}] -----------------------------")
        correct = 0
        total_loss = 0

        for i in range(len(images)):
            image = np.array(images[i]).reshape(1, -1)
            label = labels[i]

            Z1, A1, Z2 = forward_propagation(image, W1, b1, W2, b2)
            probs = softmax(Z2)
            loss = cross_entropy_loss(probs, label)
            total_loss += loss

            prediction = np.argmax(Z2)

            if prediction == label:
                correct += 1
                W2[:, label] += learning_rate * A1.flatten() * 0.1
                b2[0, label] += learning_rate * 0.1
            else:
                W2[:, label] += learning_rate * A1.flatten()
                b2[0, label] += learning_rate
                W2[:, prediction] -= learning_rate * A1.flatten()
                b2[0, prediction] -= learning_rate

        accuracy = (correct / len(images)) * 100
        avg_loss = total_loss / len(images)
        epoch_accuracies.append(accuracy)
        epoch_losses.append(avg_loss)

        print(f"  --> Précision entraînement : {accuracy:.2f}% ({correct}/{len(images)})")
        print(f"  --> Perte moyenne (cross-entropy) : {avg_loss:.4f}")

    print("\n[OK] Entraînement terminé.")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ Temps total d'exécution : {elapsed:.2f} secondes")
    return W1, b1, W2, b2, epoch_accuracies, epoch_losses

# === Étape 5 : Évaluation sur les données de test ===
def evaluate(images, labels, W1, b1, W2, b2):
    print("\n[ÉTAPE 5] Évaluation du modèle sur les données de test...")
    correct = 0
    for i in range(len(images)):
        image = np.array(images[i]).reshape(1, -1)
        _, _, Z2 = forward_propagation(image, W1, b1, W2, b2)
        prediction = np.argmax(Z2)
        if prediction == labels[i]:
            correct += 1

    accuracy = (correct / len(images)) * 100
    print(f"[RESULTAT FINAL] Précision sur le jeu de test : {accuracy:.2f}% ({correct}/{len(images)})")
    return accuracy

# === Affichage des prédictions pour les N premières images ===
def show_predictions_grid(images, labels, W1, b1, W2, b2, count=50):
    print(f"[INFO] Affichage des {count} premières prédictions ...")
    plt.figure(figsize=(14, 12))
    for i in range(count):
        image = np.array(images[i]).reshape(1, -1)
        _, _, Z2 = forward_propagation(image, W1, b1, W2, b2)
        prediction = np.argmax(Z2)

        plt.subplot(int(np.ceil(np.sqrt(count))), int(np.ceil(np.sqrt(count))), i + 1)
        plt.imshow(np.array(images[i]).reshape(28, 28), cmap='gray')
        color = "green" if prediction == labels[i] else "red"
        plt.title(f"P:{prediction} | R:{labels[i]}", color=color, fontsize=9)
        plt.axis('off')

    plt.suptitle(f"{count} premières prédictions du jeu de test", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def plot_accuracies(accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
    plt.title("Précision par époque")
    plt.xlabel("Époque")
    plt.ylabel("Précision (%)")
    plt.grid(True)
    plt.show()

def plot_losses(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o', color='red')
    plt.title("Perte moyenne (Cross-Entropy) par époque")
    plt.xlabel("Époque")
    plt.ylabel("Perte")
    plt.grid(True)
    plt.show()

# === Programme principal ===
def main():
    print("=== PERCEPTRON MULTICOUCHE POUR MNIST ===\n")

    # Chargement et prétraitement
    train_images, train_labels = load_data('./MNIST/mnist_handwritten_train.json')
    test_images, test_labels = load_data('./MNIST/mnist_handwritten_test.json')

    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)

    # Définition de l’architecture
    input_size = 784     # 28 x 28
    hidden_size = 128
    output_size = 10     # Chiffres de 0 à 9

    # Initialisation des paramètres
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    # Entraînement
    W1, b1, W2, b2, epoch_accuracies, epoch_losses = train(
        train_images, train_labels, W1, b1, W2, b2, learning_rate=0.01, epochs=10)

    # Affichage des graphiques
    plot_accuracies(epoch_accuracies)
    plot_losses(epoch_losses)

    # Évaluation
    evaluate(test_images, test_labels, W1, b1, W2, b2)

    # Affichage des 50 premières images du jeu de test
    show_predictions_grid(test_images, test_labels, W1, b1, W2, b2, count=50)

    print("\n=== FIN DU PROGRAMME ===")

if __name__ == "__main__":
    main()
