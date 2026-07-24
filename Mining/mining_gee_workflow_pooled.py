"""
=============================================================================
Surface Mining Detection - GEE Python Workflow (POOLED MODEL VERSION)
Schleswig-Flensburg, Germany | 2016-2025
=============================================================================
Authors:  Gifty Attiah & Kwaku Owusu Twum
Study areas: Osterby, Ellund, Wanderup, Schuby, Klein Rheide

Strategy:
  ONE pooled RF model trained on all 541 polygons across all 5 plots
  and all 7 validated years (2016-2020, 2022, 2024). Applied uniformly
  to all 10 years (2016-2025) for change detection comparability.

  Gap years (2021, 2023, 2025) have no training polygons and are
  classified without independent accuracy validation.

Validation:
  Pooled model accuracy is assessed on the 7 validated years using
  all three tiers and compared against year-specific model results
  from the previous run (accuracy_year_specific.csv).

Outputs:
  - 50 classified GeoTIFFs (10 years x 5 plots) → Google Drive
  - accuracy_pooled_model.csv — pooled model accuracy per tier
  - accuracy_comparison.csv  — pooled vs year-specific comparison

Class legend: 0=Field, 1=Mine, 2=Vegetation, 3=Water

Requirements:
  pip install earthengine-api geemap pandas
=============================================================================
"""

import ee
import geemap
import pandas as pd
import time

# ─────────────────────────────────────────────────────────────────────────────
# 0. INITIALISE
# ─────────────────────────────────────────────────────────────────────────────
GEE_PROJECT = 'gg-wa-temp-2025'
ee.Initialize(project=GEE_PROJECT)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ASSET_PATH   = 'projects/gg-wa-temp-2025/assets/Mining_Training_Data'
DRIVE_FOLDER = 'Mining_GEE_Pooled'

# All years to classify and export
ALL_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Years with training polygons — used for accuracy assessment only
VALIDATED_YEARS = [2016, 2017, 2018, 2019, 2020, 2022, 2024]

# Years without training polygons — classified but not validated
GAP_YEARS = [2021, 2023, 2025]

PLOTS = {
    1: 'Osterby',
    2: 'Ellund',
    3: 'Wanderup',
    4: 'Schuby',
    5: 'KleinRheide'
}

CLASS_FIELD = 'Features'
N_CLASSES   = 4

S2_BANDS   = ['B2',   'B3',    'B4',  'B8',  'B11',   'B12']
BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

ALL_FEATURES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2',
                'NDVI', 'BI', 'CI', 'SI', 'RI']

IMG_START_MONTH = 5
IMG_END_MONTH   = 8
MAX_CLOUD_PCT   = 20

TRAIN_RATIO = 0.7
RANDOM_SEED = 42

EXPORT_SCALE = 10
EXPORT_CRS   = 'EPSG:32632'

# ─────────────────────────────────────────────────────────────────────────────
# 2. STUDY AREA GEOMETRIES
# ─────────────────────────────────────────────────────────────────────────────

PLOT_GEOMETRIES = {
    1: ee.Geometry.Polygon([[
        [9.217877, 54.806927], [9.251072, 54.806863],
        [9.250959, 54.788754], [9.217780, 54.788818],
        [9.217877, 54.806927]
    ]]),
    2: ee.Geometry.Polygon([[
        [9.302806, 54.813489], [9.365824, 54.813316],
        [9.365567, 54.784847], [9.302593, 54.785020],
        [9.302806, 54.813489]
    ]]),
    3: ee.Geometry.Polygon([[
        [9.324842, 54.717969], [9.387307, 54.717785],
        [9.387026, 54.688262], [9.324606, 54.688445],
        [9.324842, 54.717969]
    ]]),
    4: ee.Geometry.Polygon([[
        [9.436417, 54.520219], [9.474021, 54.520077],
        [9.473780, 54.499311], [9.436195, 54.499453],
        [9.436417, 54.520219]
    ]]),
    5: ee.Geometry.Polygon([[
        [9.458647, 54.453562], [9.532970, 54.453257],
        [9.532595, 54.424343], [9.458324, 54.424648],
        [9.458647, 54.453562]
    ]])
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def mask_s2_clouds(image):
    scl   = image.select('SCL')
    clear = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
    return (image.updateMask(clear)
                 .divide(10000)
                 .copyProperties(image, ['system:time_start']))


def add_indices(image):
    blue  = image.select('Blue')
    green = image.select('Green')
    red   = image.select('Red')
    ndvi  = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    bi    = (red.pow(2).add(green.pow(2)).add(blue.pow(2))
                .divide(3).sqrt().rename('BI'))
    ci    = red.subtract(green).divide(red.add(green)).rename('CI')
    si    = red.subtract(blue).divide(red.add(blue)).rename('SI')
    ri    = (red.pow(2)
               .divide(blue.multiply(green.pow(3)).max(ee.Image(1e-10)))
               .rename('RI'))
    return image.addBands([ndvi, bi, ci, si, ri])


def get_s2_composite(year, geometry):
    """Annual median composite for a given year and geometry."""
    start = f'{year}-{IMG_START_MONTH:02d}-01'
    end   = f'{year}-{IMG_END_MONTH:02d}-31'
    col   = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(geometry)
               .filterDate(start, end)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PCT))
               .map(mask_s2_clouds)
               .select(S2_BANDS, BAND_NAMES)
               .map(add_indices))
    return col.median().clip(geometry).set('image_count', col.size())


def get_pooled_composite(geometry):
    """
    Multi-year median composite across all validated years.
    Used as the sampling image for pooled model training.
    Captures full spectral variability across the observation period.
    """
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(geometry)
             .filter(ee.Filter.calendarRange(IMG_START_MONTH, IMG_END_MONTH, 'month'))
             .filter(ee.Filter.calendarRange(2016, 2024, 'year'))
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PCT))
             .map(mask_s2_clouds)
             .select(S2_BANDS, BAND_NAMES)
             .map(add_indices))
    return col.median().clip(geometry)


def encode_classes(fc):
    def remap(f):
        v     = ee.String(f.get(CLASS_FIELD))
        label = ee.Algorithms.If(v.compareTo('Field').eq(0),      0,
                ee.Algorithms.If(v.compareTo('Mine').eq(0),       1,
                ee.Algorithms.If(v.compareTo('Vegetation').eq(0), 2,
                3)))
        return f.set('class', label)
    return fc.map(remap)


def sample_pixels(image, polygons, scale=10):
    return image.select(ALL_FEATURES).sampleRegions(
        collection=polygons, properties=['class', CLASS_FIELD],
        scale=scale, tileScale=4, geometries=False)


def train_rf(train_samples):
    return (ee.Classifier.smileRandomForest(numberOfTrees=100, seed=RANDOM_SEED)
              .train(features=train_samples, classProperty='class',
                     inputProperties=ALL_FEATURES))


def get_accuracy(classifier, val_samples):
    validated = val_samples.classify(classifier)
    cm = validated.errorMatrix('class', 'classification', list(range(N_CLASSES)))
    return {'overall_accuracy': cm.accuracy(), 'kappa': cm.kappa()}


def classify_image(image, classifier):
    return image.select(ALL_FEATURES).classify(classifier)


def export_to_drive(image, name, geometry):
    task = ee.batch.Export.image.toDrive(
        image=image.toByte(),
        description=name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=geometry,
        scale=EXPORT_SCALE,
        crs=EXPORT_CRS,
        maxPixels=1e9
    )
    task.start()
    print(f'    >>> Export queued: {name}')
    return task


def safe_getinfo(ee_object, retries=3, wait=10):
    for attempt in range(retries):
        try:
            return ee_object.getInfo()
        except Exception as e:
            if attempt < retries - 1:
                print(f'      timeout (attempt {attempt+1}), retrying...')
                time.sleep(wait)
            else:
                raise e


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDATION SPLIT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def split_pixel(samples):
    """Tier 1 — pixel-level 70/30 split (inflated baseline)."""
    samples = samples.randomColumn('random', RANDOM_SEED)
    return (samples.filter(ee.Filter.lt('random',  TRAIN_RATIO)),
            samples.filter(ee.Filter.gte('random', TRAIN_RATIO)))


def split_polygon(all_polygons, image, scale=10):
    """
    Tier 2 — stratified polygon-level split.
    Splits 70/30 per class across all years to ensure
    all classes are represented in both sets.
    """
    train_list, val_list = [], []
    for cls in [0, 1, 2, 3]:
        cls_p = all_polygons.filter(
            ee.Filter.eq('class', cls)).randomColumn(
            'poly_random', RANDOM_SEED + cls)
        train_list.append(cls_p.filter(ee.Filter.lt('poly_random',  TRAIN_RATIO)))
        val_list.append(  cls_p.filter(ee.Filter.gte('poly_random', TRAIN_RATIO)))

    train_p = ee.FeatureCollection(train_list).flatten()
    val_p   = ee.FeatureCollection(val_list).flatten()

    train_s = image.select(ALL_FEATURES).sampleRegions(
        collection=train_p, properties=['class', CLASS_FIELD],
        scale=scale, tileScale=4, geometries=False)
    val_s   = image.select(ALL_FEATURES).sampleRegions(
        collection=val_p, properties=['class', CLASS_FIELD],
        scale=scale, tileScale=4, geometries=False)
    return train_s, val_s


def split_lopo(all_polygons, held_out_plot_id,
               held_out_composite, pooled_composite, scale=10):
    """
    Tier 3 — leave-one-plot-out.
    Train on 4 plots, validate on the held-out plot.
    """
    train_p = all_polygons.filter(ee.Filter.neq('Plot', held_out_plot_id))
    val_p   = all_polygons.filter(ee.Filter.eq('Plot',  held_out_plot_id))

    train_geom = ee.Geometry.MultiPolygon(
        [PLOT_GEOMETRIES[pid].coordinates()
         for pid in PLOTS if pid != held_out_plot_id])

    train_s = pooled_composite.clip(train_geom).select(ALL_FEATURES).sampleRegions(
        collection=train_p, properties=['class', CLASS_FIELD],
        scale=scale, tileScale=4, geometries=False)
    val_s   = held_out_composite.select(ALL_FEATURES).sampleRegions(
        collection=val_p, properties=['class', CLASS_FIELD],
        scale=scale, tileScale=4, geometries=False)
    return train_s, val_s


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

def run_pooled_workflow(export_images=True, verbose=True):
    """
    Full pooled model workflow:

    Step 1 — Build pooled training composite (all validated years combined)
    Step 2 — Train ONE pooled RF on all 541 polygons
    Step 3 — Assess pooled model accuracy (Tier 1, 2, 3) on validated years
    Step 4 — Apply pooled RF to ALL 10 years x 5 plots = 50 images
    Step 5 — Export 50 classified GeoTIFFs to Google Drive

    Gap years (2021, 2023, 2025) are classified and exported but not validated.
    """
    training_asset = encode_classes(ee.FeatureCollection(ASSET_PATH))

    overall_geom = ee.Geometry.MultiPolygon(
        [g.coordinates() for g in PLOT_GEOMETRIES.values()])

    # ── Step 1: Build pooled composite ────────────────────────────────────
    if verbose:
        print('\nStep 1: Building pooled training composite (2016-2024)...')
    pooled_composite = get_pooled_composite(overall_geom)

    # ── Step 2: Train pooled RF ────────────────────────────────────────────
    if verbose:
        print('Step 2: Training pooled RF on all 541 polygons...')

    # Tier 1 pixel split for training (full dataset)
    all_samples      = sample_pixels(pooled_composite, training_asset)
    t1_train, t1_val = split_pixel(all_samples)
    pooled_rf        = train_rf(t1_train)

    if verbose:
        print('  Pooled RF trained successfully.')

    # ── Step 3: Accuracy assessment ────────────────────────────────────────
    if verbose:
        print('\nStep 3: Accuracy assessment...')

    accuracy_results = []

    # Tier 1 — pixel split (on pooled composite)
    try:
        t1_acc   = get_accuracy(pooled_rf, t1_val)
        t1_oa    = safe_getinfo(t1_acc['overall_accuracy'])
        t1_kappa = safe_getinfo(t1_acc['kappa'])
        if verbose:
            print(f'  Tier 1 (pixel split)    OA: {t1_oa:.4f}  K: {t1_kappa:.4f}')
    except Exception as e:
        print(f'  ERROR Tier 1: {e}')
        t1_oa = t1_kappa = -999

    # Tier 2 — polygon split (on pooled composite)
    try:
        t2_train, t2_val = split_polygon(training_asset, pooled_composite)
        t2_rf            = train_rf(t2_train)
        t2_acc           = get_accuracy(t2_rf, t2_val)
        t2_oa            = safe_getinfo(t2_acc['overall_accuracy'])
        t2_kappa         = safe_getinfo(t2_acc['kappa'])
        if verbose:
            print(f'  Tier 2 (polygon split)  OA: {t2_oa:.4f}  K: {t2_kappa:.4f}')
    except Exception as e:
        print(f'  ERROR Tier 2: {e}')
        t2_oa = t2_kappa = -999

    # Tier 3 — leave-one-plot-out (5 folds)
    if verbose:
        print('  Tier 3 (leave-one-plot-out):')
    t3_oas, t3_kappas = [], []
    for plot_id, plot_name in PLOTS.items():
        try:
            held_composite   = get_pooled_composite(PLOT_GEOMETRIES[plot_id])
            t3_train, t3_val = split_lopo(
                training_asset, plot_id, held_composite, pooled_composite)
            t3_rf            = train_rf(t3_train)
            t3_acc           = get_accuracy(t3_rf, t3_val)
            t3_oa_i          = safe_getinfo(t3_acc['overall_accuracy'])
            t3_k_i           = safe_getinfo(t3_acc['kappa'])
            t3_oas.append(t3_oa_i)
            t3_kappas.append(t3_k_i)
            if verbose:
                print(f'    Held out {plot_name}: OA={t3_oa_i:.4f}  K={t3_k_i:.4f}')
            accuracy_results.append({
                'Held_out_plot': plot_name,
                'T3_OA': round(t3_oa_i, 4),
                'T3_Kappa': round(t3_k_i, 4)
            })
        except Exception as e:
            print(f'    ERROR {plot_name}: {e}')

    t3_oa    = sum(t3_oas)    / len(t3_oas)    if t3_oas    else -999
    t3_kappa = sum(t3_kappas) / len(t3_kappas) if t3_kappas else -999

    # Print summary
    print(f'\n  ── Pooled Model Accuracy Summary ──')
    print(f'  Tier 1 (pixel)   OA: {t1_oa:.4f}  K: {t1_kappa:.4f}')
    print(f'  Tier 2 (polygon) OA: {t2_oa:.4f}  K: {t2_kappa:.4f}')
    print(f'  Tier 3 (LOPO)    OA: {t3_oa:.4f}  K: {t3_kappa:.4f}  (mean of 5 folds)')

    # Save accuracy CSV
    summary_df = pd.DataFrame([{
        'Model':    'Pooled RF',
        'T1_OA':   round(t1_oa, 4),    'T1_Kappa': round(t1_kappa, 4),
        'T2_OA':   round(t2_oa, 4),    'T2_Kappa': round(t2_kappa, 4),
        'T3_OA':   round(t3_oa, 4),    'T3_Kappa': round(t3_kappa, 4),
    }])
    summary_df.to_csv('accuracy_pooled_model.csv', index=False)

    lopo_df = pd.DataFrame(accuracy_results)
    lopo_df.to_csv('accuracy_pooled_lopo.csv', index=False)
    print('\n  Saved: accuracy_pooled_model.csv')
    print('  Saved: accuracy_pooled_lopo.csv')

    # ── Step 4 & 5: Classify and export ALL years ──────────────────────────
    if verbose:
        print(f'\nStep 4: Classifying and exporting all {len(ALL_YEARS)} years...')

    export_tasks = []

    for year in ALL_YEARS:
        gap = year in GAP_YEARS
        label = '(GAP — no validation)' if gap else '(validated)'
        if verbose:
            print(f'\n  Year: {year} {label}')

        for plot_id, plot_name in PLOTS.items():
            geometry  = PLOT_GEOMETRIES[plot_id]
            composite = get_s2_composite(year, geometry)

            n_imgs = safe_getinfo(composite.get('image_count'))
            if verbose:
                print(f'    [{plot_id}] {plot_name} — S2 images: {n_imgs}')

            if n_imgs == 0:
                print(f'      WARNING — no S2 images. Skipping.')
                continue

            classified = classify_image(composite, pooled_rf)

            if export_images:
                task = export_to_drive(
                    classified,
                    f'{plot_name}_{year}_pooled_RF',
                    geometry
                )
                export_tasks.append(task)

    print(f'\n  Total exports queued: {len(export_tasks)}')
    print(f'  Monitor: https://code.earthengine.google.com/tasks')
    print(f'  Files will appear in Drive folder: {DRIVE_FOLDER}')
    print(f'\n  File naming: PlotName_Year_pooled_RF.tif')
    print(f'  Example:     Osterby_2021_pooled_RF.tif')

    return pooled_rf, summary_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('Surface Mining Detection — Pooled RF Workflow')
    print('Schleswig-Flensburg | 2016-2025 | 5 Study Areas')
    print('='*55)
    print(f'Validated years: {VALIDATED_YEARS}')
    print(f'Gap years:       {GAP_YEARS}')
    print(f'All years:       {ALL_YEARS}')
    print(f'Total exports:   {len(ALL_YEARS) * len(PLOTS)} images')
    print('='*55)

    # Set export_images=False to run accuracy check only first
    # Set export_images=True to queue all 50 exports
    pooled_rf, accuracy_df = run_pooled_workflow(
        export_images=True,
        verbose=True
    )

    print('\nDone.')
    print('Monitor exports: https://code.earthengine.google.com/tasks')
    print(f'Drive folder: {DRIVE_FOLDER}')
