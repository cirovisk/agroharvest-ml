# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///

import matplotlib.pyplot as plt
import numpy as np

# Dados
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
B = np.array([1, 4, 9, 16, 22, 40, 50, 70, 80, 105], dtype=float)
A_smooth = np.linspace(1, 10, 100)
B_pred = 0.9697 * A_smooth + 0.667

# Configuração visual profissional (estilo acadêmico)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

# Scatter plot dos dados reais
ax.scatter(A, B, color='#1f77b4', s=60, edgecolor='black', zorder=3, label='Dados Reais (Tabela 1)')

# Linha da regressão linear simples para comparação visual
ax.plot(A_smooth, B_pred, color='#d62728', linestyle='--', linewidth=1.5, zorder=2, 
        label=r"Regressão Linear: $B'' = 0.9697A + 0.667$")

# Configurações de eixos
ax.set_title("Gráfico de Dispersão: Atributo A vs Atributo B", fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel("Atributo A (Entrada)", fontsize=10, fontweight='bold')
ax.set_ylabel("Atributo B (Saída Esperada)", fontsize=10, fontweight='bold')

ax.set_xlim(0, 11)
ax.set_ylim(0, 115)
ax.set_xticks(range(0, 12))
ax.set_yticks(range(0, 120, 10))

# Legenda e grade
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='lightgray', framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.6, color='gray')

# Salvar o gráfico
plt.tight_layout()
plt.savefig('scatter_plot.png', bbox_inches='tight')
print("Plot successfully saved as scatter_plot.png")
