## Avant

- Extraire le dossier MNIST.zip

## Fonctionnalités

- Chargement des données JSON
- Normalisation des pixels
- Réseau de neurones avec une couche cachée (ReLU)
- Entraînement avec feedback simple
- Affichage de la précision par époque
- Affichage de la fonction du coût (cross-entropy avec softmax)
  Plus la précision est haute → meilleur le modèle.
  Plus la perte est basse → meilleure la qualité des prédictions.
- Évaluation sur données de test
- Visualisation des prédictions

## Axes d'améliorations

- Ajouter la rétropropagation (Backpropagation) pour un entraînement plus efficace.
  C’est le processus qui vient après la propagation avant, où l’erreur entre la prédiction et la vérité terrain est calculée, puis utilisée pour ajuster les poids du réseau.
- Enregistrer et recharger les poids du modèle.

## Dépendances

Les dépendances nécessaires sont listées dans `requirements.txt`. Pour les installer :

```bash
pip install -r requirements.txt
```
