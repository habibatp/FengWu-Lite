import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cdsapi
import xarray as xr

# =========================
# CONFIGURATION
# =========================

PRED_DIR = "predictions"
OUT_DIR = "future_comparison_may"
os.makedirs(OUT_DIR, exist_ok=True)

# Région approximative autour de Marrakech / Maroc
# Format CDS : [North, West, South, East]
AREA = [36.0, -13.5, 27.0, -1.0]

# Point à comparer : Marrakech
TARGET_LAT = 31.63
TARGET_LON = -8.00

# Les prédictions générées :
# step 1 = +6h, ..., step 12 = +72h
STEPS = 12

# Dates ERA5 réelles à télécharger
YEAR = "2026"
MONTH = "05"
DAYS = ["01", "02", "03"]

# Ton modèle travaille en pas de 6h
TIMES = ["00:00", "06:00", "12:00", "18:00"]

# Variables pour la requête CDS (noms longs)
ERA5_REQUEST_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure"
]

# Variables dans le fichier de données (noms courts)
ERA5_DATA_VARIABLES = ["u10", "v10", "t2m", "msl"]

CHANNELS = {
    "u10": 0,
    "v10": 1,
    "t2m": 2,
    "msl": 3
}


# =========================
# FONCTIONS
# =========================

def download_era5():
    output_nc = os.path.join(OUT_DIR, "era5_real_2026_05_01_03.nc")

    if os.path.exists(output_nc):
        print(f"ERA5 déjà téléchargé : {output_nc}")
        return output_nc

    print("Téléchargement ERA5 réel 1-3 mai 2026...")

    c = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key="2b044750-3002-4014-8f68-a7fcfd7b5530"
    )

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_REQUEST_VARIABLES,
            "year": YEAR,
            "month": MONTH,
            "day": DAYS,
            "time": TIMES,
            "area": AREA,
            "format": "netcdf"
        },
        output_nc
    )

    print(f"Téléchargement terminé : {output_nc}")
    return output_nc


def nearest_grid_value(ds, var_name, time_index):
    da = ds[var_name].isel(valid_time=time_index)

    point = da.sel(
        latitude=TARGET_LAT,
        longitude=TARGET_LON,
        method="nearest"
    )

    return float(point.values)


def get_prediction_value(pred_file, var_short):
    pred = np.load(pred_file)

    ch = CHANNELS[var_short]

    # Comme ton modèle sort une grille 64x64, on prend un point proche du centre.
    # À améliorer ensuite avec conversion lat/lon -> row/col exacte.
    row = pred.shape[1] // 2
    col = pred.shape[2] // 2

    return float(pred[ch, row, col])


def main():
    era5_nc = download_era5()

    print("Lecture du fichier ERA5...")
    ds = xr.open_dataset(era5_nc)

    print(ds)

    rows = []

    for step in range(1, STEPS + 1):
        pred_file = os.path.join(
            PRED_DIR,
            f"prediction_step_{step:02d}_Tplus{step * 6}h.npy"
        )

        if not os.path.exists(pred_file):
            print(f"Prédiction absente : {pred_file}")
            continue

        # step 1 correspond au premier temps ERA5 téléchargé
        # Il faut que tes prédictions commencent bien le 1 mai 00h/06h selon ton T initial.
        time_index = step - 1

        for short in ERA5_DATA_VARIABLES:
            try:
                era5_value = nearest_grid_value(ds, short, time_index)
                pred_value = get_prediction_value(pred_file, short)

                rows.append({
                    "step": step,
                    "horizon_h": step * 6,
                    "variable": short,
                    "prediction": pred_value,
                    "era5_real": era5_value,
                    "absolute_error": abs(pred_value - era5_value)
                })

            except Exception as e:
                print(f"Erreur variable {short}, step {step}: {e}")

    df = pd.DataFrame(rows)

    csv_path = os.path.join(OUT_DIR, "comparison_prediction_vs_era5_may.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\nComparaison sauvegardée :", csv_path)
    print(df)

    for var in ["u10", "v10", "t2m", "msl"]:
        sub = df[df["variable"] == var]

        if sub.empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(sub["horizon_h"], sub["prediction"], marker="o", label="FengWu-Lite")
        plt.plot(sub["horizon_h"], sub["era5_real"], marker="o", label="ERA5 réel")
        plt.xlabel("Horizon de prévision (heures)")
        plt.ylabel(var)
        plt.title(f"Comparaison FengWu-Lite vs ERA5 réel - {var}")
        plt.legend()
        plt.grid(True)

        fig_path = os.path.join(OUT_DIR, f"comparison_{var}_may.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        print("Graphique sauvegardé :", fig_path)

    print("\n✅ Comparaison terminée.")


if __name__ == "__main__":
    main()