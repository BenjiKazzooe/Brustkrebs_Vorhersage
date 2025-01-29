# Brustkrebs-Diagnose mit Maschinellem Lernen

Dieses Projekt nutzt verschiedene Machine-Learning-Modelle zur Diagnose von Brustkrebs basierend auf dem **Breast Cancer Wisconsin (Diagnostic) Data Set**. Es werden **Logistische Regression, Support Vector Machine, Random Forest und K-Nearest Neighbors** verwendet, um die Klassifikationsleistung zu vergleichen.

## 📂 Daten
Die Daten stammen aus dem Kaggle-Datensatz: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

- **Merkmale (Features):**
  - Dies sind verschiedene diagnostische Messwerte aus einer Zellkernanalyse, die dabei helfen, zwischen gutartigen (harmlosen) und bösartigen (gefährlichen) Tumoren zu unterscheiden.
  - Wichtigste Merkmale laut Feature-Importance-Analyse:
    1. `area_worst` – Größte gemessene Zellkernfläche.
    2. `radius_worst` – Größter gemessener Radius des Zellkerns.
    3. `concave points_worst` – Anzahl der konkaven (nach innen gewölbten) Punkte des Zellkerns.
    4. `perimeter_worst` – Größter gemessener Umfang des Zellkerns.
    5. `concave points_mean` – Durchschnittliche Anzahl der konkaven Punkte.
  
  Diese Merkmale sind besonders wichtig, weil größere und unregelmäßig geformte Zellkerne oft ein Hinweis auf bösartige Tumoren sind.

- **Zielvariable (Target):**
  - `diagnosis` (B = gutartig, M = bösartig)

## 📊 Modellierung & Ergebnisse
Die Daten wurden in **80% Trainings- und 20% Testdaten** aufgeteilt und normalisiert. Für Random Forest wurde eine **Grid Search** zur Optimierung der Hyperparameter durchgeführt.

### 🔍 Confusion Matrices
Eine Confusion Matrix zeigt die Leistung eines Klassifikationsmodells, indem sie die richtigen und falschen Vorhersagen in einer Tabelle zusammenfasst.

- **True Positives (TP):** Richtig als bösartig erkannt
- **False Positives (FP):** Fälschlicherweise als bösartig erkannt
- **True Negatives (TN):** Richtig als gutartig erkannt
- **False Negatives (FN):** Fälschlicherweise als gutartig erkannt

**Logistische Regression:**
```
[[70  1]
 [ 2 41]]
```
- 70 gutartige Fälle richtig erkannt (TN)
- 1 gutartiger Fall fälschlicherweise als bösartig klassifiziert (FP)
- 2 bösartige Fälle nicht erkannt (FN)
- 41 bösartige Fälle richtig erkannt (TP)

**Support Vector Machine:**
```
[[71  0]
 [ 2 41]]
```
- Perfekte Klassifikation der gutartigen Fälle (71 TN, 0 FP)
- 2 bösartige Fälle wurden nicht erkannt (FN)
- 41 bösartige Fälle korrekt erkannt (TP)

**Random Forest:**
```
[[70  1]
 [ 3 40]]
```
- 70 gutartige Fälle richtig erkannt (TN)
- 1 gutartiger Fall fälschlicherweise als bösartig klassifiziert (FP)
- 3 bösartige Fälle nicht erkannt (FN)
- 40 bösartige Fälle richtig erkannt (TP)

**K-Nearest Neighbors:**
```
[[68  3]
 [ 3 40]]
```
- 68 gutartige Fälle richtig erkannt (TN)
- 3 gutartige Fälle fälschlicherweise als bösartig klassifiziert (FP)
- 3 bösartige Fälle nicht erkannt (FN)
- 40 bösartige Fälle richtig erkannt (TP)

### 📈 Modellevaluation
Die Leistung der Modelle wurde anhand verschiedener Metriken bewertet:

- **Train Accuracy (Trainingsgenauigkeit):** Gibt an, wie gut das Modell auf den Trainingsdaten funktioniert.
- **Test Accuracy (Testgenauigkeit):** Zeigt, wie gut das Modell auf neuen (unbekannten) Daten arbeitet.
- **Precision (Präzision):** Wie viele der vorhergesagten bösartigen Tumore tatsächlich bösartig sind.
- **Recall (Empfindlichkeit):** Wie viele der tatsächlichen bösartigen Tumore richtig erkannt wurden.
- **F1 Score:** Eine Kombination aus Präzision und Recall – ein guter Indikator für das Gleichgewicht zwischen den beiden.
- **AUC-ROC:** Zeigt, wie gut das Modell zwischen gutartigen und bösartigen Tumoren unterscheiden kann.

| Model                  | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|------------------------|---------------|--------------|-----------|--------|----------|---------|
| Logistic Regression    | 0.987         | 0.974        | 0.976     | 0.953  | 0.965    | 0.997   |
| Support Vector Machine| 0.989         | 0.982        | 1.000     | 0.953  | 0.976    | 0.997   |
| Random Forest         | 1.000         | 0.965        | 0.976     | 0.930  | 0.952    | 0.997   |
| K-Nearest Neighbors   | 0.982         | 0.947        | 0.930     | 0.930  | 0.930    | 0.981   |

## 📌 Nutzung
1. **Installiere Abhängigkeiten:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Führe das Skript aus:**
   ```bash
   python breast_cancer_classification.py
   ```

## 🛠 Technologien
- **Python (3.x)**
- `pandas`, `numpy`, `scikit-learn`
- `GridSearchCV` zur Hyperparameter-Optimierung

## 📜 Fazit
Die **Support Vector Machine (SVM)** erzielte die höchste Testgenauigkeit (98.2%) mit einer perfekten Präzision (1.0). Auch die anderen Modelle lieferten starke Ergebnisse, insbesondere **Random Forest**, das mit AUC-ROC 0.997 gut abschnitt.

---
📌 **Autor:** Sir Mölli

