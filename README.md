# Ensemble Learning for COVID-19 Risk Prediction
- implemented 7 prognosis risk prediction models for COVID-19
- introduced a competence quantification framework for assessing the competence/confidence of a model in predicting a given data entry (i.e. a digital representation of a covid patient)
- ensembled 7 prediction models for prediction using fusion strategies based on their competences
- evaluated single models and the ensembled mode on two large COVID-19 cohorts from Wuhan, China (N=2,384) and King's College Hospital (N=1,475)
# usage
1. edit the experiment setting file: `./test/test_config.json`, changing the following parameters accordingly.
  - `data_file` - the csv file which is the patient level data for prediction. *check full list of variables below*
  - (optional) `sep` - the separator, changing it to comma if your file is comma separated.
  - (optional) `mapping` - a dictionary to map column names
  - (optional) `comorbidity_cols` - the list of column names denoting comorbidities
  - 

## variables
```markdown
*demographics*
age
Male

*underline conditions*
comorbidity counts
hypertension

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