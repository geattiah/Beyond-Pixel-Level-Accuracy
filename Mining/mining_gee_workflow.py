"""
=============================================================================
Surface Mining Detection - GEE Python Workflow
Schleswig-Flensburg, Germany | 2016-2024
=============================================================================

Study areas: Osterby, Ellund, Wanderup, Schuby, Klein Rheide

Validation tiers:
  Tier 1 — Pixel-level split    (baseline, inflated, matches original study)
  Tier 2 — Polygon-level split  (publication standard, fixes autocorrelation)
  Tier 3 — Leave-one-plot-out   (transferability test, feeds Angle 2)

Requirements:
  pip install earthengine-api geemap geopandas pandas

Before running:
  1. earthengine authenticate
  2. Upload Training_Data.shp to GEE Assets
  3. Set YEARS = [2016] for a quick test before running all years
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
DRIVE_FOLDER = 'Mining_GEE_Outputs_1'

# Set to [2016] for a quick test, then restore full list
YEARS = [2016, 2017, 2018, 2019, 2020, 2022, 2024]

PLOTS = {
    1: 'Osterby',
    2: 'Ellund',
    3: 'Wanderup',
    4: 'Schuby',
    5: 'KleinRheide'
}

CLASS_FIELD    = 'Features'
N_CLASSES      = 4
# Encoding: Field=0, Mine=1, Vegetation=2, Water=3

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
    1: ee.Geometry.Polygon([[   # Osterby
        [9.217877, 54.806927], [9.251072, 54.806863],
        [9.250959, 54.788754], [9.217780, 54.788818],
        [9.217877, 54.806927]
    ]]),
    2: ee.Geometry.Polygon([[   # Ellund
        [9.302806, 54.813489], [9.365824, 54.813316],
        [9.365567, 54.784847], [9.302593, 54.785020],
        [9.302806, 54.813489]
    ]]),
    3: ee.Geometry.Polygon([[   # Wanderup
        [9.324842, 54.717969], [9.387307, 54.717785],
        [9.387026, 54.688262], [9.324606, 54.688445],
        [9.324842, 54.717969]
    ]]),
    4: ee.Geometry.Polygon([[   # Schuby
        [9.436417, 54.520219], [9.474021, 54.520077],
        [9.473780, 54.499311], [9.436195, 54.499453],
        [9.436417, 54.520219]
    ]]),
    5: ee.Geometry.Polygon([[   # Klein Rheide
        [9.458647, 54.453562], [9.532970, 54.453257],
        [9.532595, 54.424343], [9.458324, 54.424648],
        [9.458647, 54.453562]
    ]])
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def mask_s2_clouds(image):
    """Cloud/shadow masking using SCL band. Scales to 0-1."""
    scl = image.select('SCL')
    clear = (scl.neq(3).And(scl.neq(8))
                       .And(scl.neq(9))
                       .And(scl.neq(10)))
    return (image.updateMask(clear)
                 .divide(10000)
                 .copyProperties(image, ['system:time_start']))


def add_indices(image):
    """Compute 5 spectral indices. NDVI and BI are highest-importance."""
    blue  = image.select('Blue')
    green = image.select('Green')
    red   = image.select('Red')
    nir   = image.select('NIR')

    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    bi   = (red.pow(2).add(green.pow(2)).add(blue.pow(2))
               .divide(3).sqrt().rename('BI'))
    ci   = red.subtract(green).divide(red.add(green)).rename('CI')
    si   = red.subtract(blue).divide(red.add(blue)).rename('SI')
    ri   = (red.pow(2)
              .divide(blue.multiply(green.pow(3)).max(ee.Image(1e-10)))
              .rename('RI'))
    return image.addBands([ndvi, bi, ci, si, ri])


def get_s2_composite(year, geometry):
    """Median Sentinel-2 SR composite for a given year and geometry."""
    start = f'{year}-{IMG_START_MONTH:02d}-01'
    end   = f'{year}-{IMG_END_MONTH:02d}-31'
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(geometry)
             .filterDate(start, end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PCT))
             .map(mask_s2_clouds)
             .select(S2_BANDS, BAND_NAMES)
             .map(add_indices))
    composite = col.median().clip(geometry)
    return composite.set('image_count', col.size())


def encode_classes(fc):
    """
    Add integer 'class' property to each feature.
    Uses ee.String.compareTo() — .equals() is JS API only.
    Encoding: Field=0, Mine=1, Vegetation=2, Water=3
    """
    def remap(f):
        v = ee.String(f.get(CLASS_FIELD))
        label = ee.Algorithms.If(v.compareTo('Field').eq(0),      0,
                ee.Algorithms.If(v.compareTo('Mine').eq(0),       1,
                ee.Algorithms.If(v.compareTo('Vegetation').eq(0), 2,
                3)))
        return f.set('class', label)
    return fc.map(remap)


def sample_pixels(image, polygons, scale=10):
    """Sample pixel values from composite within training polygons."""
    return image.select(ALL_FEATURES).sampleRegions(
        collection=polygons,
        properties=['class', CLASS_FIELD],
        scale=scale,
        tileScale=4,
        geometries=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDATION SPLIT FUNCTIONS (all three tiers)
# ─────────────────────────────────────────────────────────────────────────────

def split_samples_pixel(samples):
    """
    Tier 1 — Pixel-level 70/30 split.
    Pixels from the same polygon can appear in both sets.
    Kept only for comparison with original study.
    WARNING: inflated accuracies from spatial autocorrelation.
    """
    samples  = samples.randomColumn('random', RANDOM_SEED)
    train    = samples.filter(ee.Filter.lt('random',  TRAIN_RATIO))
    val      = samples.filter(ee.Filter.gte('random', TRAIN_RATIO))
    return train, val


def split_samples_polygon(all_plot_polygons, image, scale=10):
    """
    Tier 2 — Stratified polygon-level split pooled across ALL YEARS.

    Why pool across years:
      Per-year polygon counts are too small for a valid split.
      Some classes (e.g. Water in Ellund, Vegetation in Schuby)
      have only 1-2 polygons per year — impossible to split 70/30.
      Pooling all 7 years gives 7-14 polygons per class, making
      a valid stratified split possible.

    Approach:
      Split polygon IDs 70/30 per class across all years.
      Train composite = median of all years (passed as `image`).
      Both train and validate sample from the same composite,
      but from spatially separate polygon sets.
    """
    classes = [0, 1, 2, 3]  # Field, Mine, Vegetation, Water
    train_polys_list = []
    val_polys_list   = []

    for cls in classes:
        cls_polys = all_plot_polygons.filter(ee.Filter.eq('class', cls))
        n_cls = cls_polys.size()
        # Only split if we have at least 2 polygons for this class
        cls_rand = cls_polys.randomColumn('poly_random', RANDOM_SEED + cls)
        train_polys_list.append(
            cls_rand.filter(ee.Filter.lt('poly_random',  TRAIN_RATIO)))
        val_polys_list.append(
            cls_rand.filter(ee.Filter.gte('poly_random', TRAIN_RATIO)))

    train_p = ee.FeatureCollection(train_polys_list).flatten()
    val_p   = ee.FeatureCollection(val_polys_list).flatten()

    train_s = image.select(ALL_FEATURES).sampleRegions(
        collection=train_p, properties=['class', CLASS_FIELD],
        scale=scale, tileScale=4, geometries=False)
    val_s   = image.select(ALL_FEATURES).sampleRegions(
        collection=val_p, properties=['class', CLASS_FIELD],
        scale=scale, tileScale=4, geometries=False)
    return train_s, val_s


def split_leave_one_plot_out(all_training, held_out_plot_id,
                             held_out_composite, all_composites,
                             scale=10):
    """
    Tier 3 — Leave-one-plot-out cross-validation.
    Trains on ALL years of the 4 non-held-out plots (maximises
    training data and avoids empty-class errors from year-only splits).
    Validates on ALL years of the held-out plot.
    Strongest test of spatial transferability. Feeds Angle 2.
    """
    train_p = all_training.filter(ee.Filter.neq('Plot', held_out_plot_id))
    val_p   = all_training.filter(ee.Filter.eq('Plot',  held_out_plot_id))

    # Use the multi-year composite for the non-held-out plots
    train_geom = ee.Geometry.MultiPolygon(
        [PLOT_GEOMETRIES[pid].coordinates()
         for pid in PLOTS if pid != held_out_plot_id]
    )
    train_composite = all_composites.clip(train_geom)

    train_s = train_composite.select(ALL_FEATURES).sampleRegions(
        collection=train_p, properties=['class', CLASS_FIELD],
        scale=scale, tileScale=4, geometries=False)
    val_s   = held_out_composite.select(ALL_FEATURES).sampleRegions(
        collection=val_p, properties=['class', CLASS_FIELD],
        scale=scale, tileScale=4, geometries=False)
    return train_s, val_s


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLASSIFIERS AND ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def train_rf(train_samples):
    """Random Forest — 100 trees."""
    return (ee.Classifier.smileRandomForest(numberOfTrees=100, seed=RANDOM_SEED)
              .train(features=train_samples,
                     classProperty='class',
                     inputProperties=ALL_FEATURES))


def train_svm(train_samples):
    """SVM with RBF kernel. C=10, gamma=0.5 — tune if OA < 0.90."""
    return (ee.Classifier.libsvm(kernelType='RBF', gamma=0.5, cost=10)
              .train(features=train_samples,
                     classProperty='class',
                     inputProperties=ALL_FEATURES))


def get_accuracy(classifier, val_samples):
    """Confusion matrix, OA, Kappa, producer/consumer accuracies."""
    validated = val_samples.classify(classifier)
    cm = validated.errorMatrix('class', 'classification',
                               list(range(N_CLASSES)))
    return {
        'confusion_matrix': cm,
        'overall_accuracy': cm.accuracy(),
        'kappa':            cm.kappa(),
        'producers_acc':    cm.producersAccuracy(),
        'consumers_acc':    cm.consumersAccuracy()
    }


def classify_image(image, classifier):
    """Apply classifier to produce a classified image."""
    return image.select(ALL_FEATURES).classify(classifier)


def export_to_drive(image, name, geometry):
    """Export classified image to Google Drive as GeoTIFF."""
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
    return task


def safe_getinfo(ee_object, retries=3, wait=10):
    """
    Wrapper around .getInfo() with retry logic for network timeouts.
    Retries up to 3 times with 10-second waits between attempts.
    """
    for attempt in range(retries):
        try:
            return ee_object.getInfo()
        except Exception as e:
            if attempt < retries - 1:
                print(f'      getInfo timeout (attempt {attempt+1}), retrying...')
                time.sleep(wait)
            else:
                raise e


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTION A — YEAR-SPECIFIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

def run_year_specific(export_images=True, verbose=True):
    """
    Trains separate RF and SVM for each year x plot combination.
    Runs all three validation tiers per iteration.
    Exports use Tier 2 classifiers (polygon-split) as most defensible.
    """
    training_asset = encode_classes(ee.FeatureCollection(ASSET_PATH))
    results = []

    for year in YEARS:
        if verbose:
            print(f'\n{"="*55}\n  YEAR: {year}\n{"="*55}')

        for plot_id, plot_name in PLOTS.items():
            if verbose:
                print(f'\n  [{plot_id}] {plot_name}')

            polygons = training_asset.filter(
                ee.Filter.And(
                    ee.Filter.eq('Year', year),
                    ee.Filter.eq('Plot', plot_id)
                )
            )

            n = safe_getinfo(polygons.size())
            if n == 0:
                if verbose: print(f'    SKIP — no training polygons')
                continue

            geometry  = PLOT_GEOMETRIES[plot_id]
            composite = get_s2_composite(year, geometry)

            n_imgs = safe_getinfo(composite.get('image_count'))
            if verbose:
                print(f'    Polygons: {n}  |  S2 images: {n_imgs}')
            if n_imgs == 0:
                print(f'    WARNING — no S2 images. Try relaxing MAX_CLOUD_PCT.')
                continue

            # ── Tier 1: Pixel-level split ──────────────────────────────
            t1_rf_oa = t1_rf_k = t1_svm_oa = t1_svm_k = -999
            t1_rf_clf = t1_svm_clf = None
            try:
                if verbose: print(f'    Tier 1 (pixel split)...')
                samples          = sample_pixels(composite, polygons)
                t1_train, t1_val = split_samples_pixel(samples)
                t1_rf_clf        = train_rf(t1_train)
                t1_svm_clf       = train_svm(t1_train)
                t1_rf_acc        = get_accuracy(t1_rf_clf,  t1_val)
                t1_svm_acc       = get_accuracy(t1_svm_clf, t1_val)
                t1_rf_oa  = safe_getinfo(t1_rf_acc['overall_accuracy'])
                t1_rf_k   = safe_getinfo(t1_rf_acc['kappa'])
                t1_svm_oa = safe_getinfo(t1_svm_acc['overall_accuracy'])
                t1_svm_k  = safe_getinfo(t1_svm_acc['kappa'])
                if verbose:
                    print(f'      RF  OA:{t1_rf_oa:.4f}  K:{t1_rf_k:.4f}')
                    print(f'      SVM OA:{t1_svm_oa:.4f}  K:{t1_svm_k:.4f}')
            except Exception as e:
                print(f'    ERROR Tier 1: {e}')

            # ── Tier 2: Polygon-level split (pooled across all years) ───
            t2_rf_oa = t2_rf_k = t2_svm_oa = t2_svm_k = -999
            t2_rf_clf = t2_svm_clf = None
            try:
                if verbose: print(f'    Tier 2 (polygon split, all years)...')
                # Pool all years for this plot — needed because per-year
                # polygon counts (1-2 per class) are too small to split
                all_plot_polygons = training_asset.filter(
                    ee.Filter.eq('Plot', plot_id))
                # All-years median composite for this plot geometry
                plot_all_composite = (
                    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(geometry)
                    .filter(ee.Filter.calendarRange(
                        IMG_START_MONTH, IMG_END_MONTH, 'month'))
                    .filter(ee.Filter.calendarRange(2016, 2024, 'year'))
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',
                                         MAX_CLOUD_PCT))
                    .map(mask_s2_clouds)
                    .select(S2_BANDS, BAND_NAMES)
                    .map(add_indices)
                    .median()
                    .clip(geometry)
                )
                t2_train, t2_val = split_samples_polygon(
                    all_plot_polygons, plot_all_composite)
                t2_rf_clf        = train_rf(t2_train)
                t2_svm_clf       = train_svm(t2_train)
                t2_rf_acc        = get_accuracy(t2_rf_clf,  t2_val)
                t2_svm_acc       = get_accuracy(t2_svm_clf, t2_val)
                t2_rf_oa  = safe_getinfo(t2_rf_acc['overall_accuracy'])
                t2_rf_k   = safe_getinfo(t2_rf_acc['kappa'])
                t2_svm_oa = safe_getinfo(t2_svm_acc['overall_accuracy'])
                t2_svm_k  = safe_getinfo(t2_svm_acc['kappa'])
                if verbose:
                    print(f'      RF  OA:{t2_rf_oa:.4f}  K:{t2_rf_k:.4f}')
                    print(f'      SVM OA:{t2_svm_oa:.4f}  K:{t2_svm_k:.4f}')
            except Exception as e:
                print(f'    ERROR Tier 2: {e}')

            # ── Tier 3: Leave-one-plot-out ─────────────────────────────
            t3_rf_oa = t3_rf_k = t3_svm_oa = t3_svm_k = -999
            t3_rf_clf = t3_svm_clf = None
            try:
                if verbose: print(f'    Tier 3 (leave-one-plot-out)...')
                # Build a multi-year composite covering all study areas
                # Using all years maximises training data per class
                all_composite = (
                    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(ee.Geometry.MultiPolygon(
                        [PLOT_GEOMETRIES[pid].coordinates()
                         for pid in PLOTS]))
                    .filter(ee.Filter.calendarRange(
                        IMG_START_MONTH, IMG_END_MONTH, 'month'))
                    .filter(ee.Filter.calendarRange(2016, 2024, 'year'))
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',
                                         MAX_CLOUD_PCT))
                    .map(mask_s2_clouds)
                    .select(S2_BANDS, BAND_NAMES)
                    .map(add_indices)
                    .median()
                )
                t3_train, t3_val = split_leave_one_plot_out(
                    training_asset, plot_id, composite, all_composite)
                t3_rf_clf        = train_rf(t3_train)
                t3_svm_clf       = train_svm(t3_train)
                t3_rf_acc        = get_accuracy(t3_rf_clf,  t3_val)
                t3_svm_acc       = get_accuracy(t3_svm_clf, t3_val)
                t3_rf_oa  = safe_getinfo(t3_rf_acc['overall_accuracy'])
                t3_rf_k   = safe_getinfo(t3_rf_acc['kappa'])
                t3_svm_oa = safe_getinfo(t3_svm_acc['overall_accuracy'])
                t3_svm_k  = safe_getinfo(t3_svm_acc['kappa'])
                if verbose:
                    print(f'      RF  OA:{t3_rf_oa:.4f}  K:{t3_rf_k:.4f}')
                    print(f'      SVM OA:{t3_svm_oa:.4f}  K:{t3_svm_k:.4f}')
            except Exception as e:
                print(f'    ERROR Tier 3: {e}')

            results.append({
                'Year': year, 'Plot': plot_id, 'Study_Area': plot_name,
                'N_Polygons': n, 'N_S2_Images': n_imgs,
                'T1_RF_OA':  round(t1_rf_oa,  4), 'T1_RF_Kappa':  round(t1_rf_k,  4),
                'T1_SVM_OA': round(t1_svm_oa, 4), 'T1_SVM_Kappa': round(t1_svm_k, 4),
                'T2_RF_OA':  round(t2_rf_oa,  4), 'T2_RF_Kappa':  round(t2_rf_k,  4),
                'T2_SVM_OA': round(t2_svm_oa, 4), 'T2_SVM_Kappa': round(t2_svm_k, 4),
                'T3_RF_OA':  round(t3_rf_oa,  4), 'T3_RF_Kappa':  round(t3_rf_k,  4),
                'T3_SVM_OA': round(t3_svm_oa, 4), 'T3_SVM_Kappa': round(t3_svm_k, 4),
            })

            # Export using Tier 2 classifiers (most defensible for publication)
            if export_images and t2_rf_clf and t2_svm_clf:
                export_to_drive(classify_image(composite, t2_rf_clf),
                                f'{plot_name}_{year}_RF',  geometry)
                export_to_drive(classify_image(composite, t2_svm_clf),
                                f'{plot_name}_{year}_SVM', geometry)
                if verbose:
                    print(f'    Exports queued (Tier 2 classifiers)')

    # ── Summary ───────────────────────────────────────────────────────
    print(f'\n  Total results collected: {len(results)}')
    if len(results) == 0:
        print('  WARNING — no results. Check errors above.')
        return pd.DataFrame()

    df = pd.DataFrame(results)
    tc = ['Year', 'Study_Area', 'N_Polygons']

    print('\n\nACCURACY SUMMARY')
    print('='*75)

    print('\n-- Tier 1: Pixel-level split (inflated baseline) --')
    print(df[tc + ['T1_RF_OA','T1_RF_Kappa','T1_SVM_OA','T1_SVM_Kappa']
             ].to_string(index=False))

    print('\n-- Tier 2: Polygon-level split (publication standard) --')
    print(df[tc + ['T2_RF_OA','T2_RF_Kappa','T2_SVM_OA','T2_SVM_Kappa']
             ].to_string(index=False))

    print('\n-- Tier 3: Leave-one-plot-out (transferability) --')
    print(df[tc + ['T3_RF_OA','T3_RF_Kappa','T3_SVM_OA','T3_SVM_Kappa']
             ].to_string(index=False))

    valid = df[df['T1_SVM_OA'] != -999]
    if len(valid) > 0:
        t1m = valid['T1_SVM_OA'].mean()
        t2m = valid['T2_SVM_OA'].mean()
        t3m = valid['T3_SVM_OA'].mean()
        print(f'\nMean SVM OA  T1:{t1m:.4f}  T2:{t2m:.4f}  T3:{t3m:.4f}')
        print(f'Inflation T1→T2: {t1m - t2m:.4f}  (spatial autocorrelation effect)')

    df.to_csv('accuracy_year_specific.csv', index=False)
    print('\nSaved: accuracy_year_specific.csv')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. OPTION B — POOLED MODEL (ALL YEARS COMBINED)
# ─────────────────────────────────────────────────────────────────────────────

def run_pooled(export_images=True, verbose=True):
    """
    Trains ONE model on all years combined, applies to every year x plot.
    Comparing accuracy vs year-specific models answers Angle 2 directly.
    """
    if verbose:
        print(f'\n{"="*55}\n  POOLED MODEL\n{"="*55}')

    training_asset = encode_classes(ee.FeatureCollection(ASSET_PATH))

    overall_geom = ee.Geometry.MultiPolygon(
        [g.coordinates() for g in PLOT_GEOMETRIES.values()]
    )

    all_composite = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(overall_geom)
        .filter(ee.Filter.calendarRange(IMG_START_MONTH, IMG_END_MONTH, 'month'))
        .filter(ee.Filter.calendarRange(2016, 2024, 'year'))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PCT))
        .map(mask_s2_clouds)
        .select(S2_BANDS, BAND_NAMES)
        .map(add_indices)
        .median()
        .clip(overall_geom)
    )

    # Polygon-level split for pooled model (Tier 2 standard)
    train_s, val_s = split_samples_polygon(training_asset, all_composite)
    pooled_rf      = train_rf(train_s)
    pooled_svm     = train_svm(train_s)

    rf_acc  = get_accuracy(pooled_rf,  val_s)
    svm_acc = get_accuracy(pooled_svm, val_s)

    print(f'\n  Pooled RF  OA:{safe_getinfo(rf_acc["overall_accuracy"]):.4f}  '
          f'K:{safe_getinfo(rf_acc["kappa"]):.4f}')
    print(f'  Pooled SVM OA:{safe_getinfo(svm_acc["overall_accuracy"]):.4f}  '
          f'K:{safe_getinfo(svm_acc["kappa"]):.4f}')

    if export_images:
        for year in YEARS:
            for plot_id, plot_name in PLOTS.items():
                geometry  = PLOT_GEOMETRIES[plot_id]
                composite = get_s2_composite(year, geometry)
                n_imgs    = safe_getinfo(composite.get('image_count'))
                if n_imgs == 0:
                    continue
                export_to_drive(classify_image(composite, pooled_rf),
                                f'{plot_name}_{year}_pooled_RF',  geometry)
                export_to_drive(classify_image(composite, pooled_svm),
                                f'{plot_name}_{year}_pooled_SVM', geometry)
                if verbose:
                    print(f'  Exports queued: {plot_name} {year}')

    return pooled_rf, pooled_svm


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('Surface Mining Detection — GEE Python Workflow')
    print('Schleswig-Flensburg | 2016-2024 | 5 Study Areas')
    print('='*55)

    # ── Tip: set YEARS = [2016] at the top for a quick test first ────

    # Option A: year-specific models with all three validation tiersd
    # Set export_images=False until accuracy looks good, then True
    results_df = run_year_specific(export_images=True, verbose=True)

    # Option B: pooled model — compare accuracy with Option A for Angle 2
    # pooled_rf, pooled_svm = run_pooled(export_images=False, verbose=True)

    print('\nDone. Monitor exports: https://code.earthengine.google.com/tasks')
    print(f'Drive folder: {DRIVE_FOLDER}')
