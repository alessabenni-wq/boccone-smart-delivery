
# ── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(2024)


# ── Generazione dati simulati ──────────────────────────────────────────────────

n = 300

complessita = np.random.uniform(1, 10, n)
carico      = np.random.uniform(1, 10, n)
distanza    = np.random.uniform(0.5, 12, n)
traffico    = np.random.uniform(1, 10, n)
pioggia     = np.random.choice([0, 1], n, p=[0.7, 0.3])

tempo_min = (
    5
    + 3.5  * distanza
    + 1.2  * traffico
    + 0.8  * complessita
    + 0.5  * carico
    + 4.0  * pioggia
    + np.random.normal(0, 3, n)
).clip(8, 100)


# ── DataFrame ─────────────────────────────────────────────────────────────────

df = pd.DataFrame({
    'complessita': complessita,
    'carico':      carico,
    'distanza':    distanza,
    'traffico':    traffico,
    'pioggia':     pioggia,
    'tempo_min':   tempo_min,
})

print(f"Dataset: {len(df)} ordini simulati")
print(df.describe().round(2))


# ── Analisi fattori: basso vs alto ────────────────────────────────────────────

fattori    = ['complessita', 'carico', 'distanza', 'traffico', 'pioggia']
etichette  = ['Complessità', 'Carico', 'Distanza', 'Traffico', 'Pioggia']

tempi_basso = [df[df[f] <  df[f].median()]['tempo_min'].mean() for f in fattori]
tempi_alto  = [df[df[f] >= df[f].median()]['tempo_min'].mean() for f in fattori]


# ── Grafico comparativo ────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

x         = np.arange(len(fattori))
larghezza = 0.35

b1 = ax.bar(x - larghezza/2, tempi_basso, larghezza,
            label='Valore basso', color='#2ecc71', edgecolor='white')
b2 = ax.bar(x + larghezza/2, tempi_alto,  larghezza,
            label='Valore alto',  color='#e74c3c', edgecolor='white')

for b, v in zip(list(b1) + list(b2), tempi_basso + tempi_alto):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3,
            f'{v:.0f}m', ha='center', fontsize=9, fontweight='bold')

ax.set_title('Impatto dei fattori sul tempo di consegna')
ax.set_xticks(x)
ax.set_xticklabels(etichette)
ax.set_ylabel('Tempo medio (min)')
ax.legend()

plt.tight_layout()
plt.savefig('smartfood_fattori.png', dpi=150)
plt.show()
print("Grafico salvato: smartfood_fattori.png")


# ── Modello ML: preparazione dati ─────────────────────────────────────────────

X = df[fattori]
y = df['tempo_min']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDimensioni Train: {X_train.shape}")
print(f"Dimensioni Test:  {X_test.shape}")


# ── Addestramento e valutazione ────────────────────────────────────────────────

modello = RandomForestRegressor(n_estimators=100, random_state=42)
modello.fit(X_train, y_train)

y_pred = modello.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"MAE: {mae:.1f} min")
print(f"R²:  {r2:.2f}")


# ── Scatter: reale vs predetto ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y_test, y_pred, alpha=0.4, s=25, color='#e67e22')
ax.plot([8, 100], [8, 100], 'r--', linewidth=1.5, label='Predizione perfetta')
ax.set_xlabel('Tempo reale (min)')
ax.set_ylabel('Tempo predetto (min)')
ax.set_title('Reale vs Predetto - SmartFood Delivery')
ax.text(10, 90, f'MAE={mae:.1f}min\nR2={r2:.2f}',
        fontsize=10, color='darkblue', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('smartfood_risultati.png', dpi=150)
plt.show()
print("Grafico salvato: smartfood_risultati.png")


# ── Previsione per ristoranti reali ───────────────────────────────────────────

ristoranti = [
    'Pizza Roma',
    'Sushi Garden',
    'Burger Station',
    'Trattoria Da Mario',
]

ordini_test = pd.DataFrame({
    'complessita': [7, 4, 5, 9],
    'carico':      [8, 3, 6, 7],
    'distanza':    [3.5, 1.2, 5.0, 2.8],
    'traffico':    [6, 2, 8, 5],
    'pioggia':     [1, 0, 0, 1],
}, index=ristoranti)

pred = modello.predict(ordini_test)

print("\n--- REPORT CONSEGNE ---")
for rist, t in zip(ristoranti, pred):
    stato = 'VELOCE' if t <= 25 else ('NORMALE' if t <= 45 else 'LENTO')
    print(f"  {rist:30s}  →  {t:.0f} min  ({stato})")