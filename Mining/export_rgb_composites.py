"""
=============================================================================
Export RGB True Colour Composites — All Sites and Years
Surface Mining Detection | Schleswig-Flensburg 2016–2025
=============================================================================
Exports Sentinel-2 true colour (R,G,B) median composites for all
5 study areas and all 10 years (2016–2025).

50 export tasks total → Google Drive folder: Mining_RGB_Composites

Requirements:
  pip install earthengine-api geemap
=============================================================================
"""

import ee
import time

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE
# ─────────────────────────────────────────────────────────────────────────────

GEE_PROJECT = 'gg-wa-temp-2025'
ee.Initialize(project=GEE_PROJECT)

DRIVE_FOLDER    = 'Mining_RGB_Composites'
IMG_START_MONTH = 5
IMG_END_MONTH   = 8
MAX_CLOUD_PCT   = 20
EXPORT_CRS      = 'EPSG:32632'
EXPORT_SCALE    = 10

# ─────────────────────────────────────────────────────────────────────────────
# ALL SITES AND YEARS
# ─────────────────────────────────────────────────────────────────────────────

ALL_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

PLOTS = {
    'Osterby':     ee.Geometry.Polygon([[
        [9.217877, 54.806927], [9.251072, 54.806863],
        [9.250959, 54.788754], [9.217780, 54.788818],
        [9.217877, 54.806927]
    ]]),
    'Ellund':      ee.Geometry.Polygon([[
        [9.302806, 54.813489], [9.365824, 54.813316],
        [9.365567, 54.784847], [9.302593, 54.785020],
        [9.302806, 54.813489]
    ]]),
    'Wanderup':    ee.Geometry.Polygon([[
        [9.324842, 54.717969], [9.387307, 54.717785],
        [9.387026, 54.688262], [9.324606, 54.688445],
        [9.324842, 54.717969]
    ]]),
    'Schuby':      ee.Geometry.Polygon([[
        [9.436417, 54.520219], [9.474021, 54.520077],
        [9.473780, 54.499311], [9.436195, 54.499453],
        [9.436417, 54.520219]
    ]]),
    'KleinRheide': ee.Geometry.Polygon([[
        [9.458647, 54.453562], [9.532970, 54.453257],
        [9.532595, 54.424343], [9.458324, 54.424648],
        [9.458647, 54.453562]
    ]])
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def mask_s2_clouds(image):
    scl   = image.select('SCL')
    clear = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
    return (image.updateMask(clear)
                 .divide(10000)
                 .copyProperties(image, ['system:time_start']))


def get_rgb_composite(year, geometry):
    """
    Build median RGB composite (Red=B4, Green=B3, Blue=B2).
    May-August window, cloud filtered.
    """
    start = f'{year}-{IMG_START_MONTH:02d}-01'
    end   = f'{year}-{IMG_END_MONTH:02d}-31'

    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(geometry)
             .filterDate(start, end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PCT))
             .map(mask_s2_clouds)
             .select(['B4', 'B3', 'B2'], ['Red', 'Green', 'Blue']))

    composite = col.median().clip(geometry)
    n_images  = col.size().getInfo()
    return composite, n_images


def stretch_rgb(image):
    """
    Linear stretch for natural-looking true colour.
    Clamps 0-0.3 reflectance range → rescales to 0-255 uint8.
    """
    return (image
            .clamp(0, 0.3)
            .divide(0.3)
            .multiply(255)
            .toByte())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

print('Exporting RGB True Colour Composites — All Sites and Years')
print('='*55)
print(f'Sites: {list(PLOTS.keys())}')
print(f'Years: {ALL_YEARS}')
print(f'Total exports: {len(ALL_YEARS) * len(PLOTS)}')
print(f'Drive folder: {DRIVE_FOLDER}')
print('='*55)

tasks   = []
skipped = []

for year in ALL_YEARS:
    print(f'\n  Year: {year}')
    for name, geometry in PLOTS.items():

        print(f'    {name}...', end=' ')
        composite, n_imgs = get_rgb_composite(year, geometry)

        if n_imgs == 0:
            print(f'SKIP — no S2 images')
            skipped.append(f'{name}_{year}')
            continue

        print(f'{n_imgs} images', end=' ')

        rgb_vis = stretch_rgb(composite)

        task = ee.batch.Export.image.toDrive(
            image=rgb_vis,
            description=f'{name}_{year}_RGB',
            folder=DRIVE_FOLDER,
            fileNamePrefix=f'{name}_{year}_RGB',
            region=geometry,
            scale=EXPORT_SCALE,
            crs=EXPORT_CRS,
            maxPixels=1e9
        )
        task.start()
        tasks.append(f'{name}_{year}_RGB')
        print(f'→ queued')

print(f'\n{"="*55}')
print(f'Export tasks queued: {len(tasks)}')
if skipped:
    print(f'Skipped (no images): {skipped}')
print(f'\nMonitor: https://code.earthengine.google.com/tasks')
print(f'Files will appear in Drive folder: {DRIVE_FOLDER}')
print('\nOnce downloaded, consolidate with consolidate_exports.py')
print('then compare visually to pick best sites/years for the figure.')
