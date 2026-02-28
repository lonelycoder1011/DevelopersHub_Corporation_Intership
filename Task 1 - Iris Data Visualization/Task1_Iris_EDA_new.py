# ═══════════════════════════════════════════════════════════════
# Task 1: Iris Dataset — Exploratory Data Analysis
# All plots saved as t1p1.png → t1p6.png
# ═══════════════════════════════════════════════════════════════

# ── Core Libraries ────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ── Global Plot Aesthetics ────────────────────────────────────
sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.0)
plt.rcParams['figure.dpi']       = 130
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize']   = 10
plt.rcParams['axes.labelsize']   = 9
plt.rcParams['xtick.labelsize']  = 8
plt.rcParams['ytick.labelsize']  = 8
plt.rcParams['legend.fontsize']  = 8

print('All libraries imported successfully!')

# ─────────────────────────────────────────────────────────────
# LOAD & INSPECT DATASET
# ─────────────────────────────────────────────────────────────
df = sns.load_dataset('iris')

print(f'\nDataset Shape : {df.shape[0]} rows x {df.shape[1]} columns')
print(f'Columns       : {df.columns.tolist()}')
print(f'\nFirst 5 Rows:\n{df.head()}')
print('\nDataset Info:')
df.info()
print(f'\nDescriptive Statistics:\n{df.describe().round(2)}')
print(f'\nSpecies Distribution:\n{df["species"].value_counts()}')
print('\nPerfectly balanced - 50 samples per class.')
print(f'\nMissing Values:\n{df.isnull().sum()}')
print('\nNo missing values - dataset is clean!')

features     = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
colors       = sns.color_palette('Set1', 3)
species_list = df['species'].unique()

# ─────────────────────────────────────────────────────────────
# HELPER: Safe main title that never gets clipped
# Uses suptitle + subplots_adjust instead of y > 1.0
# ─────────────────────────────────────────────────────────────
def add_main_title(fig, title, top=0.91):
    """
    Draws the main figure title safely inside the canvas.
    'top' sets how far down the subplots start (leaves room for title above).
    Always use bbox_inches='tight' when saving to be safe.
    """
    fig.suptitle(title, fontsize=11, fontweight='bold',
                 x=0.5, y=0.97, va='top', ha='center')
    fig.subplots_adjust(top=top)


# ═══════════════════════════════════════════════════════════════
# t1p6 - PAIRPLOT
# ═══════════════════════════════════════════════════════════════
g = sns.pairplot(
    df, hue='species', diag_kind='kde',
    plot_kws={'alpha': 0.6, 's': 35, 'edgecolor': 'white', 'linewidth': 0.4},
    diag_kws={'fill': True, 'alpha': 0.35},
    height=1.7, aspect=1.05
)
g.fig.set_size_inches(10, 9)
g.fig.suptitle(
    'Pairplot - All Feature Relationships Across All Species',
    fontsize=11, fontweight='bold', x=0.5, y=0.99, va='top'
)
g.fig.subplots_adjust(top=0.94, hspace=0.25, wspace=0.25)
plt.savefig('t1p6.png', bbox_inches='tight', dpi=130)
plt.show()
print('t1p6.png saved')


# ═══════════════════════════════════════════════════════════════
# t1p5 - SCATTER PLOTS
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
add_main_title(fig,
    'Scatter Plots - Petal Length vs Width  &  Sepal Length vs Width',
    top=0.87)

sns.scatterplot(
    data=df, x='petal_length', y='petal_width',
    hue='species', style='species', s=75,
    palette='Set1', edgecolor='white', linewidth=0.5, ax=axes[0]
)
axes[0].set_title('Petal Length vs Petal Width', pad=7)
axes[0].set_xlabel('Petal Length (cm)')
axes[0].set_ylabel('Petal Width (cm)')
axes[0].legend(title='Species', title_fontsize=8, loc='upper left', framealpha=0.8)

sns.scatterplot(
    data=df, x='sepal_length', y='sepal_width',
    hue='species', style='species', s=75,
    palette='Set1', edgecolor='white', linewidth=0.5, ax=axes[1]
)
axes[1].set_title('Sepal Length vs Sepal Width', pad=7)
axes[1].set_xlabel('Sepal Length (cm)')
axes[1].set_ylabel('Sepal Width (cm)')
axes[1].legend(title='Species', title_fontsize=8, loc='upper right', framealpha=0.8)

plt.savefig('t1p5.png', bbox_inches='tight', dpi=130)
plt.show()
print('t1p5.png saved')


# ═══════════════════════════════════════════════════════════════
# t1p4 - HISTOGRAMS + KDE
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
add_main_title(fig,
    'Histograms + KDE - Feature Value Distributions by Species',
    top=0.91)
axes = axes.flatten()

for i, feature in enumerate(features):
    ax = axes[i]
    for j, sp in enumerate(species_list):
        subset = df[df['species'] == sp][feature]
        ax.hist(subset, bins=13, alpha=0.45, color=colors[j],
                label=sp, edgecolor='white', linewidth=0.4)
        # label='_nolegend_' hides KDE line from legend - only species names show
        subset.plot.kde(ax=ax, color=colors[j], linewidth=1.8, label='_nolegend_')

    ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', pad=6)
    ax.set_xlabel(f'{feature.replace("_", " ").title()} (cm)', labelpad=3)
    ax.set_ylabel('Frequency', labelpad=3)
    ax.legend(title='Species', title_fontsize=8, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.22)

fig.subplots_adjust(hspace=0.38, wspace=0.30)
plt.savefig('t1p4.png', bbox_inches='tight', dpi=130)
plt.show()
print('t1p4.png saved')


# ═══════════════════════════════════════════════════════════════
# t1p3 - BOX PLOTS
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
add_main_title(fig,
    'Box Plots - Feature Distributions & Outlier Detection  (Red Dots = Outliers)',
    top=0.91)
axes = axes.flatten()

for i, feature in enumerate(features):
    ax = axes[i]
    sns.boxplot(
        data=df, x='species', y=feature, palette='Set2',
        width=0.42, linewidth=1.2,
        flierprops=dict(marker='o', markerfacecolor='red', markersize=5,
                        linestyle='none', markeredgecolor='darkred'),
        ax=ax
    )
    sns.stripplot(
        data=df, x='species', y=feature,
        color='black', alpha=0.22, size=2.5, jitter=True, ax=ax
    )
    ax.set_title(f'{feature.replace("_", " ").title()} by Species', pad=6)
    ax.set_xlabel('Species', labelpad=3)
    ax.set_ylabel(f'{feature.replace("_", " ").title()} (cm)', labelpad=3)
    ax.grid(True, alpha=0.22, axis='y')
    ax.tick_params(axis='x', labelsize=8)

fig.subplots_adjust(hspace=0.38, wspace=0.30)
plt.savefig('t1p3.png', bbox_inches='tight', dpi=130)
plt.show()
print('t1p3.png saved')


# ═══════════════════════════════════════════════════════════════
# t1p2 - CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════
corr = df[features].corr().round(2)

fig, ax = plt.subplots(figsize=(7, 6))
add_main_title(fig,
    'Correlation Heatmap - Linear Relationships Between All 4 Features',
    top=0.88)

sns.heatmap(
    corr, annot=True, fmt='.2f', cmap='RdYlGn',
    vmin=-1, vmax=1, linewidths=0.6, linecolor='white',
    square=True, ax=ax, annot_kws={'size': 11, 'weight': 'bold'}
)
ax.set_xticklabels([f.replace('_', '\n') for f in features], rotation=0, fontsize=8)
ax.set_yticklabels([f.replace('_', ' ').title() for f in features], rotation=0, fontsize=8)

plt.savefig('t1p2.png', bbox_inches='tight', dpi=130)
plt.show()
print('t1p2.png saved')


# ═══════════════════════════════════════════════════════════════
# t1p1 - VIOLIN PLOTS
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
add_main_title(fig,
    'Violin Plots - Petal Length & Sepal Width Distribution Shape by Species',
    top=0.87)

sns.violinplot(
    data=df, x='species', y='petal_length',
    palette='Set1', inner='quartile', linewidth=1.2, ax=axes[0]
)
axes[0].set_title('Petal Length Distribution', pad=7)
axes[0].set_xlabel('Species', labelpad=3)
axes[0].set_ylabel('Petal Length (cm)', labelpad=3)
axes[0].tick_params(axis='x', labelsize=8)

sns.violinplot(
    data=df, x='species', y='sepal_width',
    palette='Set2', inner='quartile', linewidth=1.2, ax=axes[1]
)
axes[1].set_title('Sepal Width Distribution', pad=7)
axes[1].set_xlabel('Species', labelpad=3)
axes[1].set_ylabel('Sepal Width (cm)', labelpad=3)
axes[1].tick_params(axis='x', labelsize=8)

plt.savefig('t1p1.png', bbox_inches='tight', dpi=130)
plt.show()
print('t1p1.png saved')

print('\nAll 6 plots generated and saved successfully!')
