# 🚴 SmartFood Delivery — Previsione Tempi di Consegna

> **Progetto RisaVet · Corso di Intelligenza Artificiale · Aragona**

---

## Descrizione

**SmartFood Delivery** è un sistema di previsione dei tempi di consegna basato su Machine Learning.  
A partire da dati simulati (distanza, traffico, carico, complessità ordine, condizioni meteo), il sistema:

1. genera un dataset di **300 ordini simulati** con variabili biometriche e ambientali,
2. analizza l'**impatto di ogni fattore** sul tempo di consegna tramite grafico comparativo,
3. addestra un modello **Random Forest Regressor** per predire i tempi,
4. produce un **report per ristoranti reali** con classificazione VELOCE / NORMALE / LENTO.

---

## Struttura del codice

```
smartfood_delivery.py
│
├── Generazione dati simulati          (NumPy — seed 2024)
├── Calcolo tempo di consegna          (formula pesata + rumore)
├── DataFrame e statistiche            (Pandas)
├── Analisi fattori basso vs alto      (mediana come soglia)
├── Grafico comparativo                → smartfood_fattori.png
├── Preparazione dati ML               (train/test split 80/20)
├── Addestramento Random Forest        (100 alberi)
├── Valutazione: MAE e R²
├── Scatter reale vs predetto          → smartfood_risultati.png
└── Report consegne per ristoranti
```

---

## Variabili del dataset

| Variabile     | Descrizione                        | Range        |
|---------------|------------------------------------|--------------|
| `complessita` | Complessità dell'ordine            | 1 – 10       |
| `carico`      | Carico di lavoro del corriere      | 1 – 10       |
| `distanza`    | Distanza di consegna (km)          | 0.5 – 12     |
| `traffico`    | Livello di traffico                | 1 – 10       |
| `pioggia`     | Condizioni meteo (0=sereno, 1=pioggia) | 0 / 1    |
| `tempo_min`   | Tempo di consegna (minuti) — target| 8 – 100      |

---

## Formula del tempo di consegna

```
tempo = 5
      + 3.5 × distanza
      + 1.2 × traffico
      + 0.8 × complessita
      + 0.5 × carico
      + 4.0 × pioggia
      + rumore(0, 3)
```

La distanza e il traffico hanno il peso maggiore; la pioggia aggiunge mediamente **4 minuti**.

---

## Classificazione consegne

| Tempo predetto | Stato   |
|----------------|---------|
| ≤ 25 min       | 🟢 VELOCE  |
| 26 – 45 min    | 🟡 NORMALE |
| > 45 min       | 🔴 LENTO   |

---

## Output generati

| File                      | Contenuto                                          |
|---------------------------|----------------------------------------------------|
| `smartfood_fattori.png`   | Grafico comparativo basso/alto per ogni fattore    |
| `smartfood_risultati.png` | Scatter plot reale vs predetto con MAE e R²        |
| Output terminale          | Statistiche, metriche modello, report ristoranti   |

---

## Metriche del modello

| Metrica | Significato                                  | Risultato tipico |
|---------|----------------------------------------------|------------------|
| **MAE** | Errore medio assoluto in minuti              | ~3.7 min         |
| **R²**  | Proporzione di varianza spiegata (0→1)       | ~0.89            |

---

## Requisiti

```
python >= 3.8
numpy
pandas
matplotlib
scikit-learn
```

Installazione:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Esecuzione

```bash
python smartfood_delivery.py
```

Output atteso (esempio):

```
Dataset: 300 ordini simulati
Grafico salvato: smartfood_fattori.png
Dimensioni Train: (240, 5)
Dimensioni Test:  (60, 5)
MAE: 3.7 min
R²:  0.89
Grafico salvato: smartfood_risultati.png

--- REPORT CONSEGNE ---
  Pizza Roma                      →  37 min  (NORMALE)
  Sushi Garden                    →  23 min  (VELOCE)
  Burger Station                  →  40 min  (NORMALE)
  Trattoria Da Mario              →  29 min  (NORMALE)
```

---

## Tecnologie utilizzate

| Libreria       | Utilizzo                                        |
|----------------|-------------------------------------------------|
| `NumPy`        | Generazione dati simulati, operazioni vettoriali|
| `Pandas`       | Creazione e analisi del DataFrame               |
| `Matplotlib`   | Grafici a barre e scatter plot                  |
| `Scikit-learn` | Random Forest, train/test split, MAE, R²        |

---
