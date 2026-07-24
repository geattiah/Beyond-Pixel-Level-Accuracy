"""
=============================================================================
Generate area_stats.csv from Classified GeoTIFFs
Surface Mining Detection | Schleswig-Flensburg 2016–2025
=============================================================================
Reads all classified pooled RF GeoTIFFs and computes per-class area
statistics for each study area and year.

Input:  C:\\Mining\\Mining_Classified_Pooled\\  (folder of classified tifs)
Output: area_stats.csv

Class legend: 0=Field, 1=Mine, 2=Vegetation, 3=Water

Requirements:
  pip install rasterio numpy pandas
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import rasterio

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FOLDER = r'C:\Mining\Mining_GEE_Pooled_v4'
OUTPUT_CSV   = r'C:\Mining\Mining_Results_Pooled\area_stats.csv'

PLOTS = ['Osterby', 'Ellund', 'Wanderup', 'Schuby', 'KleinRheide']
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

CLASSIFIER    = 'pooled_RF'
PIXEL_AREA_HA = (10 * 10) / 10000  # 10m resolution = 0.01 ha per pixel

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE AREA STATS
# ─────────────────────────────────────────────────────────────────────────────

records = []

for plot in PLOTS:
    for year in YEARS:
        fname = os.path.join(INPUT_FOLDER, f'{plot}_{year}_{CLASSIFIER}.tif')

        if not os.path.exists(fname):
            print(f'  MISSING: {os.path.basename(fname)}')
            continue

        with rasterio.open(fname) as src:
            arr    = src.read(1).astype(float)
            nodata = src.nodata

        # Mask nodata values
        if nodata is not None:
            arr = np.where(arr == nodata, -1, arr)

        # Count pixels per class
        total_px = int(np.sum(arr >= 0))
        field_px = int(np.sum(arr == 0))
        mine_px  = int(np.sum(arr == 1))
        veg_px   = int(np.sum(arr == 2))
        water_px = int(np.sum(arr == 3))

        # Convert to hectares
        total_ha = round(total_px * PIXEL_AREA_HA, 2)
        field_ha = round(field_px * PIXEL_AREA_HA, 2)
        mine_ha  = round(mine_px  * PIXEL_AREA_HA, 2)
        veg_ha   = round(veg_px   * PIXEL_AREA_HA, 2)
        water_ha = round(water_px * PIXEL_AREA_HA, 2)

        # Percentage of total valid area
        mine_pct  = round(mine_ha  / total_ha * 100, 2) if total_ha > 0 else 0
        water_pct = round(water_ha / total_ha * 100, 2) if total_ha > 0 else 0

        records.append({
            'Plot':     plot,
            'Year':     year,
            'Mine_ha':  mine_ha,
            'Mine_pct': mine_pct,
            'Water_ha': water_ha,
            'Water_pct':water_pct,
            'Veg_ha':   veg_ha,
            'Field_ha': field_ha,
            'Total_ha': total_ha
        })

        print(f'  {plot} {year}: '
              f'Mine={mine_ha:.2f}ha ({mine_pct}%)  '
              f'Water={water_ha:.2f}ha ({water_pct}%)')

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f'\nSaved: {OUTPUT_CSV}')
print(f'Total rows: {len(df)} ({len(PLOTS)} plots x {len(YEARS)} years)')
print(f'\nPreview:')
print(df.head(10).to_string(index=False))
