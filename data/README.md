# Dataset instructions / 数据说明

No measured or literature-derived data are included in this preparation package. / 本包不含真实论文数据。

For training, the application requires the first 15 predictor columns and the target column below. Batch prediction requires the 15 predictors; the target is not required. Headers are case-sensitive. Save as UTF-8 CSV. Do not rename CSV headers to manuscript abbreviations.

| Column | Meaning | Unit/type |
|---|---|---|
| Soil_pH | Soil pH | dimensionless |
| Soil_Clay | Soil clay content | % |
| Soil_SOM | Soil organic matter | g/kg |
| Soil_CEC | Soil cation exchange capacity | cmol/kg (as labeled in source; verify charge basis) |
| Soil_Moisture | Soil moisture | % (document measurement basis) |
| BC_Rate | Biochar application rate | % (document mass basis) |
| BC_Temp | Pyrolysis temperature | °C |
| BC_SSA | Biochar specific surface area | m²/g |
| BC_N | Biochar nitrogen content | % |
| BC_Ash | Biochar ash content | % |
| Metal_Total | Initial total metal(loid) concentration | mg/kg |
| Metal_Available | Initial available metal(loid) concentration before biochar application | mg/kg |
| BC_Feedstock | Biochar feedstock | categorical text |
| Metal_Type | Metal(loid) identity | categorical text |
| Soil_Type | Soil type | categorical text |
| Immobilization_Efficiency | Target immobilization efficiency | % on a 0–100 scale |
| Study_ID | Traceable original study identifier; required for study-wise analysis | metadata |
| Source_Title | Original source title; optional | metadata |

The current source validates the response against 0–100. If genuine mobilization records produce negative efficiencies, review and report their treatment rather than silently changing them to zero. This packaging task does not modify that behavior.

Document the exact response calculation and reference/control definition, extraction methods for available metals, dry/wet mass basis, missing-value treatment and source citations. Keep Metal_Available as the INITIAL available concentration, not the post-treatment concentration.

If the dataset cannot be distributed here, replace this note with its actual access location and terms. The built-in synthetic dataset is separate and only supports functional testing.
