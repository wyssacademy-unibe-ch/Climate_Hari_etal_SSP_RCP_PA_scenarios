# Climate_Hari_etal_inprep_Nature_Sustainability

Code supporting the manuscript:

**“The roles of climate mitigation, sustainable land use and area-based conservation in curbing future biodiversity loss”**

This repository contains the computational workflow used to project terrestrial vertebrate richness under alternative climate, land-use and protected-area (PA) scenarios, under two management assumptions, to 2080.

## 1. Analysis overview

We combined climate-driven species distribution models for 15,494 terrestrial vertebrates with LUH2 land-use projections and three PA configurations:

1. Existing 17% PA coverage
2. 30% biodiversity-only expansion
3. 30% multi-objective expansion

The analysis includes three pathways:

1. **Sustainability:** SSP1–RCP2.6
2. **Intermediate:** SSP2–RCP4.5 land use combined with RCP2.6 climate
3. **Inequality:** SSP4–RCP6.0

Projected changes were evaluated relative to a harmonised baseline representing 2015 land use and 17% PA coverage.

Species-specific climate suitability under limited dispersal was weighted by the fraction of suitable land use in each grid cell using a Land-Use Filter. The filtered occurrence probabilities were summed across species to produce continuous estimates of richness for amphibians, birds and mammals.

Two PA management-effectiveness assumptions were evaluated:

- **Full effectiveness:** all projected land-use change is prevented within the protected fraction.
- **Limited effectiveness:** 35.95% of projected land-use change is prevented within the existing 17% network and 23.59% within both 30% configurations, based on coverage-matched funding ratios.

## 2. Repository structure

### `preprocessing/`

Scripts for preparing LUH2 land-use data, PA masks and management-effectiveness scenarios. Numbered scripts should be run in ascending order.

1. **`0_create_bio-only_mask.py`**  
   Processes the Jung et al. priority rasters and creates the biodiversity-only 0.5° proportional mask.

2. **`0_create_multi-objective_mask.py`**  
   Processes the Jung et al. priority rasters and creates the biodiversity-only 0.5° proportional mask.

3. **`0_find_endemics.py`**
    Matches the IUCN country-endemic species list to the species represented in the SDM dataset, including documented taxonomic synonyms. The output provides the species subset used in the endemic sensitivity analysis.

4. **`1_17pa_2015_2080_management.py`**
    Adjusts projected LUH2 change considering the existing 17% PA network under the full- and limited-effectiveness assumptions for the present-day and future scenarios.

5. **`1_30pa_2080_management.py`**
    Adjusts projected LUH2 change considering the biodiversity-only and the multi-objective expansion scenarios under the full- and limited-effectiveness assumptions for the future scenarios.


### `processing/`

Contains separate pipelines for the full global species set (/Global) and the country-endemic subset (/Endemics).

These scripts:
**`2_luf_sumprob_PA_management.py`**

  1. Read climate-driven SDM projections.
  2. Apply the limited-dispersal constraint.
  3. Link species-specific IUCN habitat preferences to LUH2 classes.
  4. Calculate the fraction of suitable land use in each grid cell.
  5. Apply the Land-Use Filter.
  6. Sum filtered occurrence probabilities within each taxonomic group.
  7. Generate outputs for each GCM, SDM, pathway, PA configuration and management-effectiveness assumption combination.

**`3_create_ensembles.py`**

  Construct ensemble richness estimates across model combinations per taxonomic group and total vertebrate richness.

### `processing/.bash/`

Shell scripts that automate model runs across combinations of:

- Three SSP–RCP pathways
- Four GCMs: GFDL-ESM2M, IPSL-CM5A-LR, HadGEM2-ES and MIROC5
- Two SDMs: GAM and GBM
- Three taxonomic groups: amphibians, birds and mammals
- Three PA configurations: 17%-PA, biodiversity-only, multi-objective
- Full- and limited-effectiveness assumptions

1. **`run_luf_17_management.sh`**
    It launches the complete workflow to produce the outputs for the 17%-PA configuration for present-day and future scenarios.

2. **`run_luf_30_management.sh`**
    It launches the complete workflow to produce the outputs for the both 30% expansion configurations under the different future scenarios.

### `postprocessing/`

Scripts inside Jupyter notebooks used to aggregate model outputs, calculate summary statistics and generate the main and supplementary figures and tables.

Files ending in:

- **`_global.ipynb`** analyse the full species set.
- **`_endemics.ipynb`** analyse the country-endemic subset.


## 3. Recommended execution order

1. Download the needed inputs from the original sources.
2. Construct the existing and expanded PA masks.
3. Generate full- and limited-effectiveness land-use scenarios.
4. Match the country-endemic species subset.
5. Apply the Land-Use Filter to species-level SDM outputs.
6. Sum occurrence probabilities by taxonomic group.
7. Construct ensemble richness estimates.
8. Run global, subregional and taxon-specific analyses to generate figures and supplementary tables.

## 4. Technical requirements

- Python 3
- Bash
- Jupyter Notebook

Principal Python libraries:

- `xarray`
- `pandas`
- `numpy`
- `geopandas`
- `rasterio`
- `matplotlib`

## 5. Input datasets

### Species distribution models

Hof, C. et al. Bioenergy cropland expansion may offset positive effects of climate change mitigation for global vertebrate diversity. *Proceedings of the National Academy of Sciences* **115**, 13294–13299 (2018).

### Land-use projections

Hurtt, G. C. et al. Harmonization of global land use change and management for the period 850–2100 (LUH2) for CMIP6. *Geoscientific Model Development* **13**, 5425–5464 (2020).

### Existing Protected-area network

UNEP-WCMC & IUCN. Protected Planet: The World Database on Protected Areas (WDPA) (UNEP-WCMC & IUCN, 20236). Available at Protected Planet.

### Protected-area expansion configurations

Jung, M. et al. Areas of global importance for conserving terrestrial biodiversity, carbon and water. *Nature Ecology & Evolution* **5**, 1499–1509 (2021).

## 6. Outputs

The workflow produces:

- Taxon-specific summed occurrence probabilities
- Global and country-endemic richness ensembles
- Cell-level and IPBES-subregional richness changes
- Driver-attribution summaries
- Taxon-specific comparisons
- Management-effectiveness comparisons
- Main and supplementary figures and tables