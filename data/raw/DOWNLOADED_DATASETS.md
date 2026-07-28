# Downloaded Public Tabular Datasets

These files were added to increase real trajectory volume for the ARIA agentic
pipeline. They are small public tabular datasets from auth-free sources and are
intended for pipeline/eval diversity, not domain-specific product claims.

Sources used:

- MachineLearningMastery/Jason Brownlee dataset mirror:
  - `pima_indians_diabetes.csv`
  - `haberman_survival.csv`
  - `breast_cancer_wisconsin.csv`
- UCI Machine Learning Repository:
  - `heart_disease_cleveland.csv`
  - `parkinsons_voice.csv`
  - `parkinsons_telemonitoring.csv`
  - `blood_transfusion.csv`
  - `fertility_diagnosis.csv`
  - `indian_liver_patient.csv`
  - `wine_quality_red.csv`
  - `wine_quality_white.csv`
- Vanderbilt Department of Biostatistics:
  - `bodyfat_vanderbilt.csv`
- Sutanoy/Public-Regression-Datasets GitHub mirror:
  - `bone_mineral_density.csv`
  - `heart_public_regression.csv`

Some source files were headerless; headers were added during normalization so
`run_one_loop.py` can profile them consistently.
