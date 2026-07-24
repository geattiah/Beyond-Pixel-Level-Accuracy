"""
=============================================================================
Post-Processing — Majority Filter for Classified GeoTIFFs
Surface Mining Detection | Schleswig-Flensburg 2016–2025
=============================================================================
Applies a majority (mode) filter and minimum mapping unit (MMU) filter
to remove salt-and-pepper noise from classified GeoTIFFs.

Fix: uses 255 as internal nodata sentinel (valid for uint8) instead of -1.

Input:  Mining_Classified_Pooled  (raw classified GeoTIFFs)
Output: Mining_Classified_Cleaned (cleaned GeoTIFFs, same filenames)

Requirements:
  pip install rasterio numpy scipy
=============================================================================
"""

import os
import glob
import numpy as np
import rasterio
from scipy import ndimage

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FOLDER  = r'C:\Mining\Mining_Classified_Pooled'
OUTPUT_FOLDER = r'C:\Mining\Mining_Classified_Pooled_Cleaned'

FILTER_SIZE     = 3     # 3x3 majority filter window
APPLY_TWICE     = True  # Apply filter twice for stronger smoothing
MIN_MINE_PIXELS = 9     # Minimum Mine patch size (9 px = 0.09 ha at 10m)

# Internal sentinel for nodata — must be valid for uint8 (0-255)
# 255 is safe since our class values are 0,1,2,3 only
NODATA_SENTINEL = 255

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def fast_majority_filter(arr, size=3):
    """
    Majority (mode) filter using scipy generic_filter.
    Pixels with value NODATA_SENTINEL are excluded from neighbourhood
    calculation and preserved as-is in output.
    """
    def mode_func(values):
        valid = values[values != NODATA_SENTINEL]
        if len(valid) == 0:
            return NODATA_SENTINEL
        return np.bincount(valid.astype(int)).argmax()

    work     = arr.copy().astype(np.uint8)
    filtered = ndimage.generic_filter(
        work.astype(float),
        mode_func,
        size=size,
        mode='nearest'
    ).astype(np.uint8)

    # Preserve original nodata locations
    filtered[arr == NODATA_SENTINEL] = NODATA_SENTINEL
    return filtered


def remove_small_mine_patches(arr, min_pixels=9):
    """
    Remove Mine (value=1) patches smaller than min_pixels.
    Reclassifies them as the most common neighbouring non-Mine class.
    """
    result    = arr.copy()
    mine_mask = (arr == 1)
    labeled, n_patches = ndimage.label(mine_mask)

    removed = 0
    for i in range(1, n_patches + 1):
        patch      = labeled == i
        patch_size = int(patch.sum())

        if patch_size < min_pixels:
            # Find surrounding non-Mine, non-nodata pixels
            dilated     = ndimage.binary_dilation(patch, iterations=2)
            border_mask = (dilated & ~patch &
                           (arr != 1) & (arr != NODATA_SENTINEL))
            border_vals = arr[border_mask]

            if len(border_vals) > 0:
                replacement = int(np.bincount(
                    border_vals.astype(int)).argmax())
            else:
                replacement = 2  # Default to Vegetation

            result[patch] = replacement
            removed += patch_size

    return result, n_patches, removed


def process_image(input_path, output_path):
    """Apply filters to a single classified GeoTIFF."""
    fname = os.path.basename(input_path)
    print(f'\n  Processing: {fname}')

    with rasterio.open(input_path) as src:
        arr    = src.read(1).astype(np.uint8)
        meta   = src.meta.copy()
        nodata = src.nodata

    # Map original nodata to our internal sentinel
    arr_work = arr.copy()
    if nodata is not None:
        arr_work[arr == int(nodata)] = NODATA_SENTINEL

    original_mine_px = int(np.sum(arr_work == 1))
    print(f'    Original Mine pixels: {original_mine_px}')

    # Step 1a: Majority filter
    print(f'    Step 1a: Majority filter ({FILTER_SIZE}x{FILTER_SIZE})...')
    arr_filtered = fast_majority_filter(arr_work, size=FILTER_SIZE)

    # Step 1b: Second pass
    if APPLY_TWICE:
        print(f'    Step 1b: Second majority filter pass...')
        arr_filtered = fast_majority_filter(arr_filtered, size=FILTER_SIZE)

    after_filter = int(np.sum(arr_filtered == 1))
    print(f'    After filter: {after_filter} Mine pixels '
          f'({original_mine_px - after_filter} removed)')

    # Step 2: MMU filter
    print(f'    Step 2: MMU filter (min {MIN_MINE_PIXELS} px = '
          f'{MIN_MINE_PIXELS * 0.01:.2f} ha)...')
    arr_cleaned, n_patches, mmu_removed = remove_small_mine_patches(
        arr_filtered, min_pixels=MIN_MINE_PIXELS)

    final_mine_px = int(np.sum(arr_cleaned == 1))
    print(f'    Mine patches found: {n_patches}')
    print(f'    After MMU: {final_mine_px} Mine pixels '
          f'({mmu_removed} removed)')
    print(f'    Total removed: {original_mine_px - final_mine_px} pixels '
          f'({(original_mine_px - final_mine_px)/original_mine_px*100:.1f}%)')

    # Restore original nodata
    arr_out = arr_cleaned.copy()
    if nodata is not None:
        arr_out[arr_cleaned == NODATA_SENTINEL] = int(nodata)

    # Save — keep same dtype (uint8) and metadata
    meta.update({'dtype': 'uint8', 'nodata': nodata})
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(arr_out, 1)

    print(f'    Saved: {os.path.basename(output_path)}')
    return original_mine_px, final_mine_px


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('Post-Processing: Majority Filter + MMU Filter')
    print('='*55)
    print(f'Input:       {INPUT_FOLDER}')
    print(f'Output:      {OUTPUT_FOLDER}')
    print(f'Filter size: {FILTER_SIZE}x{FILTER_SIZE}  (applied twice: {APPLY_TWICE})')
    print(f'Min Mine:    {MIN_MINE_PIXELS} px = {MIN_MINE_PIXELS*0.01:.2f} ha')
    print('='*55)

    tif_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*.tif')))
    print(f'\nFound {len(tif_files)} GeoTIFFs to process')

    if len(tif_files) == 0:
        print('No files found. Check INPUT_FOLDER path.')
        exit()

    results = []
    for input_path in tif_files:
        output_path = os.path.join(OUTPUT_FOLDER,
                                   os.path.basename(input_path))
        try:
            orig, clean = process_image(input_path, output_path)
            pct = round((orig - clean) / orig * 100, 1) if orig > 0 else 0
            results.append((os.path.basename(input_path),
                            orig, clean, orig-clean, pct))
        except Exception as e:
            print(f'    ERROR: {e}')

    print(f'\n{"="*55}')
    print(f'COMPLETE — {len(results)} files processed')
    print(f'{"="*55}')
    print(f'\n{"File":<42} {"Orig":>7} {"Clean":>7} {"Rmvd":>7} {"%":>6}')
    print('-'*68)
    for fname, orig, clean, rmvd, pct in results:
        print(f'{fname:<42} {orig:>7} {clean:>7} {rmvd:>7} {pct:>5}%')

    print(f'\nCleaned files saved to: {OUTPUT_FOLDER}')
    print('\nNext: update CLASSIFIED_FOLDER in fig_classification_maps.py')
    print(f'      to: {OUTPUT_FOLDER}')
