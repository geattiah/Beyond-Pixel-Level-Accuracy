# Beyond Pixel-Level Accuracy in Sentinel-2 Surface Mining Classification Across a 10-Year Time Series

**Author:** Gifty Attiah 

---

## Overview

This repository contains the classified outputs and training polygons for a 10-year Sentinel-2 surface mining classification and change detection study across five sand and gravel extraction sites in Schleswig-Flensburg, Germany (2016–2025).

The study introduces a three-tier spatial validation framework that quantifies accuracy inflation from spatial autocorrelation and assesses geographic transferability of Random Forest and Support Vector Machine classifiers.

---

## Repository Contents

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

## Citation

Attiah, G. (2025). Beyond Pixel-Level Accuracy in Sentinel-2 Surface Mining Classification Across a 10-Year Time Series. 

---

## Data

Sentinel-2 imagery accessed via the `COPERNICUS/S2_SR_HARMONIZED` collection in Google Earth Engine. 

---

## License

Code is released under the MIT License. Classified output GeoTIFFs are released under CC BY 4.0.
