"""
=============================================================================
Phase 5 — Change Detection & Trajectory Analysis
Schleswig-Flensburg Surface Mining | 2016-2024
=============================================================================
Inputs:  Classified GeoTIFFs from Phase 4 (RF classifier)
         Located in: C:\\Mining\\Mining_Classified\\

Outputs:
  1. mined_area_stats.csv      — mined area (km2) and % per plot per year
  2. water_area_stats.csv      — site water area per plot per year
  3. change_matrix.csv         — year-to-year land cover transitions
  4. site_trajectories.csv     — trajectory class per plot (expanding etc.)
  5. Figures saved to:         C:\\Mining\\Mining_Figures\\
       - mined_area_trends.png     (10-year trajectory per plot)
       - water_area_trends.png
       - change_heatmap.png        (annual change matrix visualisation)
       - trajectory_summary.png    (bar chart of trajectory types)

Class legend (from GEE classification):
  0 = Field
  1 = Mine
  2 = Vegetation
  3 = Water

Requirements:
  pip install rasterio numpy pandas matplotlib seaborn geopandas
=============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import rasterio
from rasterio.transform import from_bounds
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FOLDER  = r'C:\Mining\Mining_GEE_Pooled_Cleaned_v4'
OUTPUT_FOLDER = r'C:\Mining\Mining_Figures_Pooled'
CSV_FOLDER    = r'C:\Mining\Mining_Results_Pooled'

# Pooled RF outputs — all years use same classifier
CLASSIFIER = 'pooled_RF'

PLOTS = ['Osterby', 'Ellund', 'Wanderup', 'Schuby', 'KleinRheide']
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Gap years — classified by pooled model but no independent validation
# Flagged visually in all figures with open markers and dashed lines
GAP_YEARS = [2021, 2023, 2025]

# Class encoding
CLASS_FIELD      = 1
CLASS_WATER      = 3
CLASS_VEGETATION = 2
CLASS_BACKGROUND = 0

# Class labels for display
CLASS_LABELS = {0: 'Field', 1: 'Mine', 2: 'Vegetation', 3: 'Water'}
CLASS_COLORS = {0: '#F4A460', 1: '#8B0000', 2: '#228B22', 3: '#4169E1'}

# Pixel area in km2 at 10m resolution
PIXEL_AREA_KM2 = (10 * 10) / 1e6  # = 0.0001 km2

# Trajectory classification thresholds
EXPAND_THRESHOLD   =  0.05   # >5% increase = Expanding
DECLINE_THRESHOLD  = -0.05   # >5% decrease = Reclaiming
STABLE_THRESHOLD   =  0.05   # within ±5%  = Stable

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER,    exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_classified_image(plot, year, classifier=CLASSIFIER):
    """Load a classified GeoTIFF and return array + pixel count."""
    fname = os.path.join(INPUT_FOLDER, f'{plot}_{year}_{classifier}.tif')
    if not os.path.exists(fname):
        print(f'  MISSING: {os.path.basename(fname)}')
        return None, None, None
    with rasterio.open(fname) as src:
        arr      = src.read(1)
        meta     = src.meta
        nodata   = src.nodata
    # Mask nodata
    if nodata is not None:
        arr = np.where(arr == nodata, -1, arr)
    return arr, meta, fname


def count_pixels(arr, class_val):
    """Count pixels of a given class, excluding nodata (-1)."""
    return int(np.sum(arr == class_val))


def total_valid_pixels(arr):
    """Count all valid (non-nodata) pixels."""
    return int(np.sum(arr >= 0))


def pixels_to_km2(n_pixels):
    return round(n_pixels * PIXEL_AREA_KM2, 4)


def compute_change_matrix(arr_before, arr_after):
    """
    Compute a 4x4 land cover transition matrix.
    Rows = from class, Cols = to class.
    Only counts valid pixels in both images.
    """
    valid = (arr_before >= 0) & (arr_after >= 0)
    matrix = np.zeros((4, 4), dtype=int)
    for from_cls in range(4):
        for to_cls in range(4):
            matrix[from_cls, to_cls] = int(
                np.sum((arr_before == from_cls) &
                       (arr_after  == to_cls)  & valid))
    return matrix


def classify_trajectory(areas_km2):
    """
    Classify a site's mining trajectory over the observation period.

    Categories:
      Expanding          — net increase >5% AND increases in ≥60% of transitions
      Reclaiming         — net decrease >5% AND decreases in ≥60% of transitions
      Stable             — net change within ±5%
      Fluctuating-Rising — mixed trend but net increase >5%
                           (active extraction with seasonal variability)
      Fluctuating-Declining — mixed trend but net decrease >5%
                           (possible wind-down with interruptions)

    Note: Sand and gravel mining is inherently cyclical — annual composites
    capture seasonal expansion and contraction. Fluctuating patterns are
    expected and reflect operational rather than directional change.
    """
    if len(areas_km2) < 2:
        return 'Insufficient data'

    values  = [a for a in areas_km2 if a is not None]
    if len(values) < 2:
        return 'Insufficient data'

    net_change  = values[-1] - values[0]
    pct_change  = net_change / values[0] if values[0] > 0 else 0

    diffs       = [values[i+1] - values[i] for i in range(len(values)-1)]
    n_increases = sum(1 for d in diffs if d > 0)
    n_decreases = sum(1 for d in diffs if d < 0)
    pct_increasing = n_increases / len(diffs)
    pct_decreasing = n_decreases / len(diffs)

    if pct_change > EXPAND_THRESHOLD and pct_increasing >= 0.6:
        return 'Expanding'
    elif pct_change < DECLINE_THRESHOLD and pct_decreasing >= 0.6:
        return 'Reclaiming'
    elif abs(pct_change) <= STABLE_THRESHOLD:
        return 'Stable'
    elif pct_change > EXPAND_THRESHOLD:
        return 'Fluctuating-Rising'
    elif pct_change < DECLINE_THRESHOLD:
        return 'Fluctuating-Declining'
    else:
        return 'Stable'


# ─────────────────────────────────────────────────────────────────────────────
# 3. AREA STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_area_stats():
    """
    Compute mined area and site water area for each plot x year.
    Returns two DataFrames: mined_df and water_df.
    """
    mined_records = []
    water_records = []

    print('Computing area statistics...')

    for plot in PLOTS:
        for year in YEARS:
            arr, meta, fpath = load_classified_image(plot, year)
            if arr is None:
                continue

            total_px   = total_valid_pixels(arr)
            mine_px    = count_pixels(arr, CLASS_FIELD)       # Mine = 1
            # Note: CLASS_FIELD=1 is Mine in our encoding
            mine_px    = count_pixels(arr, 1)   # Mine
            water_px   = count_pixels(arr, 3)   # Water
            veg_px     = count_pixels(arr, 2)   # Vegetation
            field_px   = count_pixels(arr, 0)   # Field/bare

            total_area = pixels_to_km2(total_px)
            mine_area  = pixels_to_km2(mine_px)
            water_area = pixels_to_km2(water_px)
            veg_area   = pixels_to_km2(veg_px)
            field_area = pixels_to_km2(field_px)

            mine_pct   = round(mine_area  / total_area * 100, 2) if total_area > 0 else 0
            water_pct  = round(water_area / total_area * 100, 2) if total_area > 0 else 0

            mined_records.append({
                'Plot': plot, 'Year': year,
                'Mine_km2': mine_area,  'Mine_pct': mine_pct,
                'Water_km2': water_area,'Water_pct': water_pct,
                'Veg_km2': veg_area,    'Field_km2': field_area,
                'Total_km2': total_area
            })

            print(f'  {plot} {year}: Mine={mine_area:.4f}km2 ({mine_pct}%)  '
                  f'Water={water_area:.4f}km2 ({water_pct}%)')

    mined_df = pd.DataFrame(mined_records)
    mined_df.to_csv(os.path.join(CSV_FOLDER, 'area_stats.csv'), index=False)
    print(f'\nSaved: area_stats.csv')
    return mined_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_change_detection():
    """
    Compute year-to-year land cover transition matrices for each plot.
    Year pairs: 2016-2017, 2017-2018, 2018-2019, 2019-2020,
                2020-2022, 2022-2024
    """
    year_pairs = [(YEARS[i], YEARS[i+1]) for i in range(len(YEARS)-1)]
    all_changes = []

    print('\nComputing change matrices...')

    for plot in PLOTS:
        print(f'\n  {plot}:')
        for y1, y2 in year_pairs:
            arr1, _, _ = load_classified_image(plot, y1)
            arr2, _, _ = load_classified_image(plot, y2)

            if arr1 is None or arr2 is None:
                continue

            # Resize to same shape if needed
            if arr1.shape != arr2.shape:
                min_rows = min(arr1.shape[0], arr2.shape[0])
                min_cols = min(arr1.shape[1], arr2.shape[1])
                arr1 = arr1[:min_rows, :min_cols]
                arr2 = arr2[:min_rows, :min_cols]

            matrix = compute_change_matrix(arr1, arr2)

            # Mine gain/loss
            mine_gain = sum(matrix[c][CLASS_FIELD]
                           for c in range(4) if c != CLASS_FIELD)
            mine_loss = sum(matrix[CLASS_FIELD][c]
                           for c in range(4) if c != CLASS_FIELD)
            mine_net  = (mine_gain - mine_loss) * PIXEL_AREA_KM2

            print(f'    {y1}→{y2}: Mine gain={pixels_to_km2(mine_gain):.4f}km2  '
                  f'loss={pixels_to_km2(mine_loss):.4f}km2  '
                  f'net={mine_net:.4f}km2')

            # Store flattened matrix
            for from_cls in range(4):
                for to_cls in range(4):
                    all_changes.append({
                        'Plot': plot,
                        'Year_from': y1, 'Year_to': y2,
                        'From_class': CLASS_LABELS[from_cls],
                        'To_class':   CLASS_LABELS[to_cls],
                        'Pixels':     matrix[from_cls][to_cls],
                        'Area_km2':   pixels_to_km2(matrix[from_cls][to_cls])
                    })

    change_df = pd.DataFrame(all_changes)
    change_df.to_csv(os.path.join(CSV_FOLDER, 'change_matrix.csv'), index=False)
    print(f'\nSaved: change_matrix.csv')
    return change_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAJECTORY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_trajectories(area_df):
    """
    Classify each plot's mining trajectory over the full period.
    Also computes net change and percentage change.
    """
    print('\nClassifying trajectories...')
    records = []

    for plot in PLOTS:
        sub = area_df[area_df['Plot'] == plot].sort_values('Year')
        if len(sub) == 0:
            continue

        areas      = sub['Mine_km2'].tolist()
        years      = sub['Year'].tolist()
        trajectory = classify_trajectory(areas)
        net_change = areas[-1] - areas[0] if len(areas) >= 2 else 0
        pct_change = (net_change / areas[0] * 100) if areas[0] > 0 else 0

        print(f'  {plot}: {trajectory}  '
              f'({areas[0]:.4f} → {areas[-1]:.4f} km2, '
              f'{pct_change:+.1f}%)')

        records.append({
            'Plot':            plot,
            'Trajectory':      trajectory,
            'Mine_start_km2':  round(areas[0],  4) if len(areas) > 0 else None,
            'Mine_end_km2':    round(areas[-1],  4) if len(areas) > 0 else None,
            'Mine_peak_km2':   round(max(areas), 4) if len(areas) > 0 else None,
            'Peak_year':       years[areas.index(max(areas))] if len(areas) > 0 else None,
            'Net_change_km2':  round(net_change, 4),
            'Pct_change':      round(pct_change, 2),
            'N_years':         len(areas),
            'Start_year':      years[0]  if len(years) > 0 else None,
            'End_year':        years[-1] if len(years) > 0 else None,
        })

    traj_df = pd.DataFrame(records)
    traj_df.to_csv(os.path.join(CSV_FOLDER, 'trajectories.csv'), index=False)
    print(f'\nSaved: trajectories.csv')
    return traj_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def plot_mined_area_trends(area_df):
    """
    Figure 1 — 10-year mined area trajectory per plot.
    Line chart with mine area (km2) on y-axis, year on x-axis.
    One line per plot. Matches style of original paper Figure 9.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#264653']

    for i, plot in enumerate(PLOTS):
        sub = area_df[area_df['Plot'] == plot].sort_values('Year')

        # Validated years — solid line and filled markers
        validated = sub[~sub['Year'].isin(GAP_YEARS)]
        gap       = sub[sub['Year'].isin(GAP_YEARS)]

        ax.plot(validated['Year'], validated['Mine_km2'],
                marker='o', linewidth=2, markersize=7,
                color=colors[i], label=plot)

        # Gap years — open markers and dashed segment
        if len(gap) > 0:
            ax.plot(gap['Year'], gap['Mine_km2'],
                    marker='o', linewidth=1.5, markersize=7,
                    color=colors[i], linestyle='--',
                    markerfacecolor='white', markeredgewidth=2)

        # Annotate last point
        if len(sub) > 0:
            last = sub.iloc[-1]
            ax.annotate(f'{last.Mine_km2:.3f}',
                        (last.Year, last.Mine_km2),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=8, color=colors[i])

    # Add note about gap years
    ax.axvspan(2020.5, 2021.5, alpha=0.08, color='grey')
    ax.axvspan(2022.5, 2023.5, alpha=0.08, color='grey')
    ax.text(2021, ax.get_ylim()[0], '†', ha='center', fontsize=10, color='grey')
    ax.text(2023, ax.get_ylim()[0], '†', ha='center', fontsize=10, color='grey')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Mined Area (km²)', fontsize=12)
    ax.set_title('Mined Area Extent 2016–2024\nSchleswig-Flensburg, Germany',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(YEARS)
    ax.legend(title='Study Area', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, 'mined_area_trends.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: mined_area_trends.png')


def plot_water_area_trends(area_df):
    """
    Figure 2 — Site water area trends per plot.
    Water expansion indicates deeper excavation reaching groundwater.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#264653']

    for i, plot in enumerate(PLOTS):
        sub = area_df[area_df['Plot'] == plot].sort_values('Year')
        ax.plot(sub['Year'], sub['Water_km2'],
                marker='s', linewidth=2, markersize=7,
                color=colors[i], label=plot, linestyle='--')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Site Water Area (km²)', fontsize=12)
    ax.set_title('Site Water Area Extent 2016–2024\nSchleswig-Flensburg, Germany',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(YEARS)
    ax.legend(title='Study Area', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, 'water_area_trends.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: water_area_trends.png')


def plot_combined_trends(area_df):
    """
    Figure 3 — Combined mined area + water per plot (dual axis).
    One subplot per study area — mirrors original paper Figure 9 style.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors_mine  = '#8B0000'
    colors_water = '#4169E1'

    for i, plot in enumerate(PLOTS):
        ax = axes[i]
        sub = area_df[area_df['Plot'] == plot].sort_values('Year')

        ax2 = ax.twinx()
        l1, = ax.plot(sub['Year'],  sub['Mine_km2'],
                      marker='o', color=colors_mine,
                      linewidth=2, label='Mined area')
        l2, = ax2.plot(sub['Year'], sub['Water_km2'],
                       marker='s', color=colors_water,
                       linewidth=2, linestyle='--', label='Site water')

        ax.set_title(plot, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Mine (km²)', color=colors_mine)
        ax2.set_ylabel('Water (km²)', color=colors_water)
        ax.set_xticks(YEARS)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.2)

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)

    # Hide unused subplot
    axes[-1].set_visible(False)

    fig.suptitle('Mined Area and Site Water Trends 2016–2024',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, 'combined_trends.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: combined_trends.png')


def plot_trajectory_summary(traj_df):
    """
    Figure 4 — Bar chart summarising trajectory types across all plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: trajectory category per plot
    traj_colors = {
        'Expanding':   '#8B0000',
        'Reclaiming':  '#228B22',
        'Stable':      '#F4A460',
        'Fluctuating': '#457B9D'
    }
    bars = axes[0].barh(
        traj_df['Plot'],
        traj_df['Net_change_km2'],
        color=[traj_colors.get(t, 'grey') for t in traj_df['Trajectory']]
    )
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('Net Change in Mined Area (km²)')
    axes[0].set_title('Net Mining Change 2016–2024', fontweight='bold')

    # Add trajectory label to each bar
    for bar, (_, row) in zip(bars, traj_df.iterrows()):
        axes[0].text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            row['Trajectory'], va='center', fontsize=9
        )

    # Right: percentage change
    colors_pct = ['#8B0000' if p > 0 else '#228B22'
                  for p in traj_df['Pct_change']]
    axes[1].barh(traj_df['Plot'], traj_df['Pct_change'], color=colors_pct)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Percentage Change in Mined Area (%)')
    axes[1].set_title('Percentage Mining Change 2016–2024', fontweight='bold')

    # Legend
    patches = [mpatches.Patch(color=c, label=t)
               for t, c in traj_colors.items()]
    axes[0].legend(handles=patches, loc='lower right', fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, 'trajectory_summary.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: trajectory_summary.png')


def plot_change_heatmap(change_df):
    """
    Figure 5 — Heatmap of net mine area change per plot per year transition.
    """
    # Extract mine net change per plot per year transition
    mine_changes = change_df[
        (change_df['To_class'] == 'Mine') |
        (change_df['From_class'] == 'Mine')
    ].copy()

    # Pivot: net mine gain per transition
    year_pairs = [(YEARS[i], YEARS[i+1]) for i in range(len(YEARS)-1)]
    pair_labels = [f'{y1}→{y2}' for y1, y2 in year_pairs]

    matrix = pd.DataFrame(index=PLOTS, columns=pair_labels, dtype=float)

    for plot in PLOTS:
        for y1, y2 in year_pairs:
            label = f'{y1}→{y2}'
            sub = change_df[
                (change_df['Plot'] == plot) &
                (change_df['Year_from'] == y1) &
                (change_df['Year_to']   == y2)
            ]
            if len(sub) == 0:
                matrix.loc[plot, label] = np.nan
                continue
            # Net mine change = pixels gained into Mine - pixels lost from Mine
            to_mine   = sub[sub['To_class']   == 'Mine']['Area_km2'].sum()
            from_mine = sub[sub['From_class'] == 'Mine']['Area_km2'].sum()
            # Subtract diagonal (mine staying mine)
            stay_mine = sub[(sub['From_class'] == 'Mine') &
                            (sub['To_class']   == 'Mine')]['Area_km2'].sum()
            net = (to_mine - stay_mine) - (from_mine - stay_mine)
            matrix.loc[plot, label] = round(net, 4)

    matrix = matrix.astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(matrix, annot=True, fmt='.4f', center=0,
                cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'Net Mine Change (km²)'})
    ax.set_title('Net Mined Area Change per Year Transition\n(red=expansion, green=reduction)',
                 fontweight='bold')
    ax.set_xlabel('Year Transition')
    ax.set_ylabel('Study Area')
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, 'change_heatmap.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: change_heatmap.png')


def plot_landcover_stacked(area_df):
    """
    Figure 6 — Stacked area chart of all 4 land cover classes per plot.
    Shows how the full landscape composition has shifted over time.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, plot in enumerate(PLOTS):
        ax  = axes[i]
        sub = area_df[area_df['Plot'] == plot].sort_values('Year')

        ax.stackplot(sub['Year'],
                     sub['Mine_km2'], sub['Water_km2'],
                     sub['Veg_km2'],  sub['Field_km2'],
                     labels=['Mine', 'Water', 'Vegetation', 'Field'],
                     colors=['#8B0000', '#4169E1', '#228B22', '#F4A460'],
                     alpha=0.8)

        ax.set_title(plot, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Area (km²)')
        ax.set_xticks(YEARS)
        ax.tick_params(axis='x', rotation=45)
        if i == 0:
            ax.legend(loc='upper left', fontsize=8)

    axes[-1].set_visible(False)
    fig.suptitle('Land Cover Composition 2016–2024',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, 'landcover_stacked.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: landcover_stacked.png')


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('Phase 5 — Change Detection & Trajectory Analysis')
    print('Schleswig-Flensburg | 2016–2024')
    print('='*55)

    # Step 1: Area statistics
    area_df   = compute_area_stats()

    # Step 2: Change detection matrices
    change_df = compute_change_detection()

    # Step 3: Trajectory classification
    traj_df   = classify_trajectories(area_df)

    # Step 4: Figures
    print('\nGenerating figures...')
    plot_mined_area_trends(area_df)
    plot_water_area_trends(area_df)
    plot_combined_trends(area_df)
    plot_trajectory_summary(traj_df)
    plot_change_heatmap(change_df)
    plot_landcover_stacked(area_df)

    # Step 5: Summary printout
    print('\n' + '='*55)
    print('TRAJECTORY SUMMARY')
    print('='*55)
    print(traj_df[['Plot','Trajectory','Mine_start_km2','Mine_end_km2',
                   'Mine_peak_km2','Peak_year',
                   'Net_change_km2','Pct_change',
                   'Start_year','End_year']].to_string(index=False))

    print(f'\nAll CSVs saved to:    {CSV_FOLDER}')
    print(f'All figures saved to: {OUTPUT_FOLDER}')
    print('\nDone.')
