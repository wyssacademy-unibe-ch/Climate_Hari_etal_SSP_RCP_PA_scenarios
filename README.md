# Climate_Hari_etal_inprep_Nature_Sustainability
Code for: "The roles of climate mitigation, sustainable land use and area-based conservation in curbing future biodiversity loss"
This repository contains the computational pipeline used to evaluate vertebrate biodiversity outcomes under three key Shared Socioeconomic Pathways: "Sustainability" (SSP1-2.6), "Middle-of-the-road" (SSP2-2.5), and "Inequality" (SSP4-6.0).1. 

1. Methodology Overview
The core analysis utilizes a Land Use Filtering (LUF) framework to project species richness.
Baseline Reference: Land-use data and Protected Area (PA) freeze logic are anchored to 2015.
Species distribution based on climate: Species Distribution Models (SDMs) from Hof et al., (2018).
Effectiveness Modeling: Comparisons are made between "full-effectiveness" and a "limited-effectiveness scenario (using an effectiveness rate of 23.59%).

2. Repository Structure

   preprocessing/   Scripts for integrating Land Use Harmonization (LUH2) data with spatial conservation targets.
     *_fractional_mask.py: Creates masks for the 30% PA expansion scenarios: biodiversity-only (bio) and multi-objective (bcw).
     *_lim_eff.py: Generates the baseline files required for the "limited-effectiveness" modeling scenario.
   find_endemics.py: Identifies and extracts scientific names for endemic vertebrates from the broader SDM pool.

   processing/   Contains distinct pipelines for Global (15,000+ species) and Endemic species analysis. Scripts used to apply the LUF approach to SDM outputs (Hof et al., 2018) and calculate the summed probability of occurrence (species richness) per taxonomic group, GCM, SDM and SSP-RCP. Ensemble files where used to calculate the total richness across the different model combinations.
    .bash/: Contains shell scripts to automate runs across all GCMs (GFDL-ESM2M, IPSL-CM5A-LR, HadGEM2-ES, MIROC5), SDMs (GBM and GAM), taxonomic group (Mammals, Bird and Amphibians) and SSP-RCP scenarios.

      results/   Jupyter Notebooks for generating manuscript figures (main text and Supplementary Information). Separated by "main" for all the species and "endemics" for the subset of endemic species.

4. Technical SpecificationsLanguage:
   Python 3.x
   Core Libraries: xarray, pandas, numpy, geopandas, rasterio, matplotlibData

5. Input Sources:
- Hof et al. (2018) and Hof, C., Voskamp, A., Biber, M. F., Böhning-Gaese, K., Engelhardt, E. K., Niamir, A., Willis, S. G., & Hickler, T. (2018). Bioenergy cropland expansion may offset positive effects of climate change mitigation for global vertebrate diversity. Proceedings of the National Academy of Sciences of the United States of America, 115(52), 13294–13299. https://doi.org/10.1073/pnas.1807745115
- Hurtt, G. C., Chini, L., Sahajpal, R., Frolking, S., Bodirsky, B. L., Calvin, K., Doelman, J. C., Fisk, J., Fujimori, S., Klein Goldewijk, K., Hasegawa, T., Havlik, P., Heinimann, A., Humpenöder, F., Jungclaus, J., Kaplan, J. O., Kennedy, J., Krisztin, T., Lawrence, D., … Zhang, X. (2020). Harmonization of global land use change and management for the period 850-2100 (LUH2) for CMIP6. Geosci. Model Dev, 13, 5425–5464. https://doi.org/10.5194/gmd-13-5425-2020
