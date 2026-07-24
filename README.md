# Beyond Pixel-Level Accuracy in Sentinel-2 Surface Mining Classification Across a 10-Year Time Series

**Author:** Gifty Attiah 

---

## Overview

This repository contains the code and classified outputs for a 10-year Sentinel-2 surface mining classification and change detection study across five sand and gravel extraction sites in Schleswig-Flensburg, Germany (2016–2025).

The study introduces a three-tier spatial validation framework that quantifies accuracy inflation from spatial autocorrelation and assesses geographic transferability of Random Forest and Support Vector Machine classifiers.

---

## Repository Contents

### Code
- `mining_gee_workflow_pooled.py` — Pooled RF model: training, three-tier validation, and classification of all 10 years × 5 study areas
- `mining_gee_workflow.py` — Year-specific RF and SVM classifiers across all 35 study area × year combinations
- `postprocess_filter.py` — Post-classification majority filter (3×3, applied twice) and MMU constraint (0.09 ha)
- `change_detection.py` — Annual transition matrices, net Mine gain/loss, and trajectory classification
- `generate_area_stats.py` — Mined area and site water area statistics (ha) per study area per year

### Classified Outputs
Pooled RF classified GeoTIFFs for all 10 years × 5 study areas (50 files total), post-processed with majority filter and MMU constraint.

- Format: GeoTIFF, EPSG:32632, 10 m resolution
- Classes: 0 = Field, 1 = Mine, 2 = Vegetation, 3 = Water
- Naming convention: `{StudyArea}_{Year}_pooled_RF.tif`

---

## Study Areas

| Code | Study Area | Area (km²) | Mining Sites |
|------|-----------|-----------|-------------|
| OS | Osterby | 4.30 | 1 |
| EL | Ellund | 12.83 | 3 |
| WA | Wanderup | 13.22 | 1 |
| SC | Schuby | 5.62 | 4 |
| KR | Klein Rheide | 15.50 | 4 |

---

## Requirements

```
earthengine-api
geemap
pandas
numpy
rasterio
matplotlib
scipy
```

Install with:
```bash
pip install earthengine-api geemap pandas numpy rasterio matplotlib scipy
```

---

## Usage

1. Authenticate with Google Earth Engine:
```bash
earthengine authenticate
```

2. Set your GEE project ID at the top of each script:
```python
GEE_PROJECT = 'your-project-id'
```

3. Run the pooled model workflow:
```bash
python mining_gee_workflow.py
```

4. Post-process classified outputs:
```bash
python postprocess_filter.py
```

5. Generate area statistics and change detection:
```bash
python generate_area_stats.py
python change_detection.py
```

---

## Citation

Attiah, G. (2025). Beyond Pixel-Level Accuracy in Sentinel-2 Surface Mining Classification Across a 10-Year Time Series. 

---

## Data

Sentinel-2 imagery accessed via the `COPERNICUS/S2_SR_HARMONIZED` collection in Google Earth Engine. Training polygon data available from the corresponding author upon reasonable request and subject to permission from DGPEIS.

---

## License

Code is released under the MIT License. Classified output GeoTIFFs are released under CC BY 4.0.
