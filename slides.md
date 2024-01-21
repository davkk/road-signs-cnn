---
marp: true
filetype: pdf
theme: uncover
class: invert
style: |
    * {
        text-align: left;
        margin-left: 0 !important;
        margin-right: 0 !important;
        width: fit-content;
    }
    .MathJax {
        margin-top: 1em !important;
        margin-bottom: 1em !important;
    }
    pre {
        width: 100%;
    }
---

# 🚗 🛑
# Klasyfikacja znaków drogowych

Dawid Karpiński, 24.01.2024 r.

---

# 1. Specyfikacja danych

---

[German Traffic Sign Dataset](https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset)

- **43** unikalnych rodzajów znaków
- **34799** zdjęć do trenowania
- **4410** zdjęć do walidacji
- **12630** zdjęć do testowania
- **każde zdjęcie ma 32x32 px**

---

![bg contain](./figures/data_sample.png)

---

Dane były *niezbalansowane*

```python
from torch.utils.data import WeightedRandomSampler

samples_weights = 1 / counts

WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
)
```

![bg contain left](./figures/data_balance.png)

---

# 2. Model: CNN

---

<!-- TODO: do dopowiedzenia opis każdej warstwy -->

```python
# ConvDownBlock:
nn.Conv2d(...) -> nn.BatchNorm2d(...) -> nn.ReLU(...) -> nn.MaxPool2d(...)

# CNN
model = nn.Sequential(
    # (*, 3, 32, 32) -> (*, 8, 16, 16)
    ConvDownBlock(3, 8, 2, 1),
    # (*, 8, 16, 16) -> (*, 16, 8, 8)
    ConvDownBlock(8, 16, 2, 1),
    # (*, 16, 8, 8) -> (*, 32, 4, 4)
    ConvDownBlock(16, 32, 2, 1),
    #
    nn.Flatten(),
    nn.Dropout(dropout),
    #
    nn.Linear(32 * 4 * 4, 128),
    nn.ReLU(),
    nn.Linear(128, len(number_of_classes)),
)
```

---

# 3. Wyniki

---

## 3.1. Wpływ dropout

---

![bg contain](./figures/loss_dropout.png)

---

![bg contain](./figures/accuracy_dropout.png)

---

## 3.2. Wpływ learning rate

---

![bg contain](./figures/loss_lr.png)

---

![bg contain](./figures/accuracy_lr_epoch.png)

---

# Dziękuję za uwagę
