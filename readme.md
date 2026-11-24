# Deep CBIR : recherche d'images par le contenu

Comparaison de deux extracteurs de descripteurs pour la recherche d'images par le contenu (CBIR) sur la base Corel (10 classes, 900 images de référence, 100 requêtes) :

1. **CNN pré-entraîné** (VGG16, ImageNet, global average pooling, 512 dim.)
2. **Autoencodeur convolutif profond** (code latent de 128 dim.)

## Méthode

- Prétraitement adapté à chaque modèle (`preprocess_input` pour VGG16, normalisation [0, 1] pour l'autoencodeur)
- Visualisation des feature maps de VGG16 (block1, block3, block5)
- Similarité cosinus et distance euclidienne
- Évaluation : Precision@k et mAP, globalement et par classe

## Résultats

| Méthode | P@1 | P@5 | P@10 | mAP |
|---|---|---|---|---|
| VGG16 - cosinus | 0.98 | 0.960 | 0.960 | 0.876 |
| VGG16 - euclidienne | 0.94 | 0.922 | 0.895 | 0.744 |
| Autoencodeur - cosinus | 0.60 | 0.504 | 0.479 | 0.319 |
| Autoencodeur - euclidienne | 0.54 | 0.464 | 0.456 | 0.316 |

**Enseignements**

- VGG16 (features sémantiques, supervisé) surpasse largement l'autoencodeur, qui encode surtout couleurs et composition globale.
- L'autoencodeur ne réussit que sur les classes homogènes (dinosaures 0.99, fleurs 0.89).
- Le cosinus est préférable à l'euclidienne pour VGG16 : après normalisation L2, les deux donnent exactement le même mAP (0.8758), ce qui montre que l'écart vient de la variation des normes des descripteurs.

## Structure attendue du dataset

Le dataset n'est pas inclus dans le dépôt.

```
DeepCBIR_Dataset/
├── training_set/<classe>/*.jpg
└── test_set/<classe>/*.jpg
```

## Lancer le notebook

```bash
pip install -r requirements.txt
jupyter notebook Project3_DeepCBIR.ipynb
```

Adapter la variable `DATA_DIR` en tête du notebook. Le notebook a été développé sous Google Colab (l'entraînement de l'autoencodeur est nettement plus rapide sur GPU).

## Limites et pistes

- Petite base d'entraînement, défavorable à l'autoencodeur
- Jeu de test utilisé pour l'early stopping et l'évaluation (effet faible)
- Pistes : autoencodeur denoising ou variationnel, fine-tuning du CNN, PCA sur les descripteurs, comparaison avec ResNet50