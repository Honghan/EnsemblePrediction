# Ensemble Learning for COVID-19 Risk Prediction
- implemented 7 prognosis risk prediction models for COVID-19
- introduced a competence quantification framework for assessing the competence/confidence of a model in predicting a given data entry (i.e. a digital representation of a covid patient)
- ensembled 7 prediction models for prediction using fusion strategies based on their competences
- evaluated single models and the ensembled mode on two large COVID-19 cohorts from Wuhan, China (N=2,384) and King's College Hospital (N=1,475)
# usage
### Prerequisite
install python libraries using `requirements.txt`
```bash
pip install -r requirements.txt
```
### Setup and Run
1. edit the experiment setting file: `./test/test_config.json`, changing the following parameters accordingly.
    - `data_file` - the csv file which is the patient level data for prediction. *check full list of variables [below](https://github.com/Honghan/EnsemblePrediction#variables)*, if numeric columns are missing, they will be automatically imputed using model related distributions.
    - (optional) `sep` - the separator, changing it to comma if your file is comma separated.
    - (optional) `mapping` - a dictionary to map column names
    - `comorbidity_cols` - the list of column names (binary valued) denoting comorbidities. NB: **if you don't have `morbidity_Hypertension` (a mandatory variable), please add an entry to `binary_columns_to_impute`. This will populate all zero column for `binary_columns_to_impute`.**
2. run the models
    ```python
    python test_util.py
    ```
3. check the result files, which are to be saved to `./test` folder.
    ```bash
    # performance tables
    death_result.tsv
    poor_prognosis_result.tsv
    
    # figures
    auc_fig_death.png
    auc_fig_poor_prognosis.png
    calibration_fig_death.png
    calibration_fig_poor_prognosis.png
    ```

## variables
```markdown
*demographics*
age
Male

*underline conditions*
morbidity_Hypertension

*bloods*
Lactate dehydrogenase
Albumin
C-reactive protein
Serum sodium
Serum blood urea nitrogen
Red cell distribution width
Lymphocyte count
Neutrophil count
Direct bilirubin

*vitals*
Oxygen saturation

*outcome (binary, 1 means event happened)*
death
poor_prognosis - defined as either death or ICU admission
```

## contact
email: honghan.wu@ucl.ac.uk
