# Importaciones necesarias
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from wordcloud import WordCloud

PROCESSED_PATH = Path("datos_procesados.parquet")
df = pd.read_parquet(PROCESSED_PATH)
df['Severity'] = pd.Categorical(df['Severity'], categories=[1, 2, 3, 4], ordered=True)

def cramers_v(contingency_table):
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# 1. Análisis de Variables Dicotómicas
dichotomous_vars = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
for var in dichotomous_vars:

    plt.figure(figsize=(8, 4))
    sns.countplot(x=var, hue='Severity', data=df, palette='viridis')
    plt.title(f'Distribución de Severity por {var}')
    plt.show()

    contingency_table = pd.crosstab(df[var], df['Severity'])
    cramers_v_value = cramers_v(contingency_table)
    print(f'{var}: Cramer\'s V = {cramers_v_value:.4f}')

# 2. Análisis de Variables Geográficas
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Start_Lng', y='Start_Lat', hue='Severity', data=df.sample(frac=0.01), palette='viridis', alpha=0.5)  # Muestreo para evitar sobrecarga
plt.title('Distribución Geográfica de Accidentes por Severity')
plt.show()

# Relación entre la distancia y la severidad
plt.figure(figsize=(8, 6))
sns.boxplot(x='Severity', y='Distance(km)', data=df, palette='viridis')
plt.title('Distribución de la Distancia por Severity')
plt.show()

# 3. Análisis de Variables Temporales
df['Hour'] = pd.to_datetime(df['Start_Time']).dt.hour

plt.figure(figsize=(10, 6))
sns.countplot(x='Hour', hue='Severity', data=df, palette='viridis')
plt.title('Distribución de Severity por Hora del Día')
plt.show()


light_conditions = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
for condition in light_conditions:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=condition, hue='Severity', data=df, palette='viridis')
    plt.title(f'Distribución de Severity por {condition}')
    plt.show()

# 4. Análisis de Variables de Tiempo y Clima
plt.figure(figsize=(8, 6))
sns.boxplot(x='Severity', y='Temperature(C)', data=df, palette='viridis')
plt.title('Distribución de la Temperatura por Severity')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Severity', y='Visibility(km)', data=df, palette='viridis')
plt.title('Distribución de la Visibilidad por Severity')
plt.show()

# 5. Análisis de Variables de Tráfico
plt.figure(figsize=(8, 6))
sns.countplot(x='Traffic_Signal', hue='Severity', data=df, palette='viridis')
plt.title('Distribución de Severity por Presencia de Señales de Tráfico')
plt.show()