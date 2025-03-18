import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
from scipy.stats import ks_2samp
from pathlib import Path
from geopandas import GeoDataFrame
import geopandas as gpd
from geopandas.tools import sjoin

INPUT_PATH = Path("accidentes.csv")
OUTPUT_PATH = Path("datos_procesados.parquet")

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

df = pd.read_csv("accidentes.csv", low_memory=False, parse_dates=["Start_Time", "End_Time", "Weather_Timestamp"])

cols_to_drop = [
    "Wind_Chill(F)", "Description", "Source", "Street", 
    "City", "Zipcode", "Country", "Airport_Code", "ID","End_Lng","End_Lat",'Precipitation(in)', "Temperature(F)", "Wind_Speed(mph)", "Visibility(mi)"
]

df["Temperature(C)"] = (df["Temperature(F)"] - 32) * 5/9
df["Wind_Speed(km/h)"] = df["Wind_Speed(mph)"] * 1.60934 
df["Visibility(km)"] = df["Visibility(mi)"] * 1.60934


if "End_Lat" in df.columns and "End_Lng" in df.columns:
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

print("\n *Resumen Estadístico de Variables Numéricas:*")
print(df.describe())

print("\n *Resumen de Variables Categóricas:*")
print(df.describe(include="object"))

missing_values = (df.isna().mean() * 100).round(1)
missing_values = missing_values[missing_values > 0]
plt.figure(figsize=(14, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, color="#1f77b4")
plt.title("Porcentaje de Valores Faltantes por Variable", fontsize=14)
plt.ylabel("Porcentaje")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


print("\n *Visualización de Valores Faltantes:*")

plt.figure(figsize=(12, 5))
msno.bar(df, color="#1f77b4")
plt.title("Cantidad de Valores Faltantes", fontsize=14)
plt.show()


variables = {
    'Timezone': 'categorical',
    'Weather_Timestamp': 'datetime',
    'Temperature(C)': 'numeric',
    'Humidity(%)': 'numeric',
    'Pressure(in)': 'numeric',
    'Visibility(km)': 'numeric',
    'Wind_Direction': 'categorical',
    'Wind_Speed(km/h)': 'numeric',
    'Weather_Condition': 'categorical',
    'Sunrise_Sunset': 'categorical',
    'Civil_Twilight': 'categorical',
    'Nautical_Twilight': 'categorical',
    'Astronomical_Twilight': 'categorical'
}
df_original = df.copy()

df.dropna(subset=['Weather_Condition'], inplace=True)

for col, col_type in variables.items():
    if col_type == 'numeric':
        df[col] = df.groupby("State")[col].transform(lambda x: x.fillna(x.median()))


for col, col_type in variables.items():
    if col_type == 'categorical':
        probs = df[col].value_counts(normalize=True) 
        df[col] = df[col].apply(lambda x: np.random.choice(probs.index, p=probs.values) if pd.isna(x) else x)


ks_results = {}
alpha = 0.05  

for col, col_type in variables.items():
    if col_type == 'numeric' and df[col].isna().sum() == 0:  
        original_values = df_original[col].dropna()
        imputed_values = df[col]

        stat, p_value = ks_2samp(original_values, imputed_values)

        ks_results[col] = {
            'KS_Statistic': stat,
            'P_Value': p_value,
            'Same_Distribution': p_value > alpha  
        }
freq_results = {}

for col, col_type in variables.items():
    if col_type == 'categorical' and col in df.columns:
        original_freq = df_original[col].value_counts(normalize=True)
        imputed_freq = df[col].value_counts(normalize=True)
        freq_results[col] = {
            'Original_Frequencies': original_freq,
            'Imputed_Frequencies': imputed_freq
        }

for col, result in ks_results.items():
    print(f"Variable: {col}")
    print(f"  KS Statistic: {result['KS_Statistic']:.4f}")
    print(f"  P-Value: {result['P_Value']:.4f}")
    print(f"  ¿Mantiene la distribución? {'Sí' if result['Same_Distribution'] else 'No'}\n")

for col, result in freq_results.items():
    print(f"Variable: {col}")
    print("  Frecuencias Originales:")
    print(result['Original_Frequencies'])
    print("  Frecuencias Imputadas:")
    print(result['Imputed_Frequencies'])
    print()

def categorize_weather(condition):
    categories = {
        "Lluvia ligera": ["Light Rain", "Light Rain Showers", "Light Rain Shower", "Light Rain / Windy"],
        "Lluvia moderada": ["Rain", "Rain Shower", "Rain Showers", "Rain Shower / Windy"],
        "Lluvia intensa": ["Heavy Rain", "Heavy Rain Showers", "Heavy Rain Shower", "Heavy Rain Shower / Windy", "Heavy Rain / Windy"],
        "Llovizna ligera": ["Light Drizzle", "Light Drizzle / Windy"],
        "Llovizna moderada": ["Drizzle", "Drizzle / Windy"],
        "Llovizna intensa": ["Heavy Drizzle"],
        "Nieve ligera": ["Light Snow", "Light Snow Showers", "Light Snow Shower", "Light Snow / Windy"],
        "Nieve moderada": ["Snow", "Snow Showers", "Snow / Windy"],
        "Nieve intensa": ["Heavy Snow", "Heavy Snow Showers", "Heavy Snow / Windy"],
        "Aguanieve ligera": ["Light Sleet", "Light Snow and Sleet", "Light Snow and Sleet / Windy"],
        "Aguanieve moderada": ["Sleet", "Snow and Sleet", "Snow and Sleet / Windy"],
        "Aguanieve intensa": ["Heavy Sleet", "Heavy Sleet / Windy"],
        "Granizo ligero": ["Light Hail", "Small Hail"],
        "Granizo moderado": ["Hail"],
        "Cielo despejado": ["Clear", "Fair"],
        "Nublado": ["Cloudy", "Mostly Cloudy", "Scattered Clouds", "Overcast", "Partly Cloudy"],
        "Tormenta eléctrica": ["Thunderstorm", "Thunder", "T-Storm", "Thunder in the Vicinity", "Thunder / Windy", "Thunder / Wintry Mix", "Light Thunderstorm"],
        "Tormentas fuertes": ["Heavy T-Storm", "Heavy Thunderstorms and Rain", "Heavy Thunderstorms and Snow", "Heavy T-Storm / Windy"],
        "Tormenta con granizo": ["Thunder and Hail", "Thunder and Hail / Windy", "Heavy Thunderstorms with Small Hail"],
        "Lluvia helada ligera": ["Light Freezing Rain", "Light Freezing Rain / Windy"],
        "Lluvia helada moderada": ["Freezing Rain", "Freezing Rain / Windy"],
        "Lluvia helada intensa": ["Heavy Freezing Rain", "Heavy Freezing Rain / Windy"],
        "Llovizna helada ligera": ["Light Freezing Drizzle"],
        "Llovizna helada moderada": ["Freezing Drizzle"],
        "Llovizna helada intensa": ["Heavy Freezing Drizzle"],
        "Niebla ligera": ["Mist", "Light Fog", "Partial Fog", "Shallow Fog", "Light Haze"],
        "Niebla densa": ["Fog", "Fog / Windy", "Patches of Fog", "Patches of Fog / Windy", "Shallow Fog / Windy", "Light Freezing Fog"],
        "Humo y neblina": ["Haze", "Smoke", "Heavy Smoke", "Haze / Windy", "Smoke / Windy"],
        "Viento fuerte": ["Blowing Dust / Windy", "Cloudy / Windy", "Fair / Windy", "Mostly Cloudy / Windy", "Partly Cloudy / Windy", "Windy", "Blowing Sand", "Widespread Dust", "Blowing Dust", "Blowing Snow Nearby", "Blowing Snow", "Sand / Windy", "Sand / Dust Whirlwinds", "Sand / Dust Whirls Nearby", "Sand / Dust Whirlwinds / Windy", "Duststorm", "Drifting Snow / Windy"],
        "Mezcla invernal": ["Wintry Mix", "Wintry Mix / Windy", "Snow and Thunder", "Snow and Thunder / Windy", "Light Snow with Thunder", "Rain and Sleet"],
        "Tornado": ["Tornado", "Funnel Cloud"],
        "Otros": ["Volcanic Ash", "Showers in the Vicinity", "Light Rain with Thunder", "Low Drifting Snow", "Light Ice Pellets", "Squalls", "N/A Precipitation", "Sand", "Snow Grains", "Heavy Ice Pellets", "Drizzle and Fog", "Light Snow Grains", "Heavy Blowing Snow", "Light Blowing Snow", "Dust Whirls", "Sleet and Thunder", "Heavy Sleet and Thunder", "Heavy Sleet / Windy"]
    }
    for category, values in categories.items():
        if condition in values:
            return category
    return "Desconocido"

def assign_weather_categories(df, column_name="Weather_Condition"):
    df[column_name] = df[column_name].apply(categorize_weather)
    return df


def clean_temperature(temp):
    if temp < -62.2:
        return np.random.uniform(-62.2, -61.9)  
    elif temp > 56.7:
        return np.random.uniform(56.5, 56.7)  
    return temp  

def clean_pressure(pres):
    if pres < 25 or pres > 32:
        return None  
    elif pres < 25.69:
        return np.random.uniform(25.69, 25.9)  
    elif pres > 31.42:
        return np.random.uniform(31.2, 31.42)  


def clean_visibility(vis):
    return np.clip(vis, 0, 10) 


df["Temperature(C)"] = df["Temperature(C)"].apply(clean_temperature)
df["Pressure(in)"] = df["Pressure(in)"].apply(clean_pressure)
df["Visibility(km)"] = df["Visibility(km)"].apply(lambda x: np.clip(x, 0, 16.09)) 


df = assign_weather_categories(df, "Weather_Condition")


df = df.dropna(subset=["Pressure(in)"])

geometry = gpd.points_from_xy(df["Start_Lng"], df["Start_Lat"])
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


states = gpd.read_file("https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/us-states.json")
states = states.to_crs("EPSG:4326")
gdf = gpd.sjoin(gdf, states[['geometry', 'name']], how='left', predicate='within')
gdf.drop(columns=['index_right'], inplace=True, errors='ignore')
gdf["Start_Time"] = pd.to_datetime(gdf["Start_Time"], errors="coerce")

if not OUTPUT_PATH.exists():
    
    gdf.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nDataset procesado guardado en: {OUTPUT_PATH}")
else:
    print("\n¡El archivo procesado ya existe! Elimínalo para regenerarlo.")