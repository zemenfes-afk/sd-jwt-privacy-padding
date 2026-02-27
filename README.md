# SD-JWT Metadata Obfuscation

## 1. Install Requirements

Install the necessary Python dependencies for cryptography, data processing, and machine learning:

```bash
pip install jwcrypto sd-jwt pandas scikit-learn

```

## 2. Generate FHIR Datasets

Generate the synthetic patient health records (e.g., using Synthea) and place the resulting JSON bundles into the input directory:

```bash
./run_synthea -p 1200
# Ensure the generated .json files are moved to output/fhir/

```

## 3. Generate Unpadded SD-JWTs

Convert the FHIR records into standard, vulnerable SD-JWTs where the disclosure count perfectly matches the active disorders:

```bash
python generate_unpadded_sdjwts.py
# Outputs saved to output/unpadded_sdjwts/
Use for both Covert_both.py
```

## 4. Generate Padded SD-JWTs

Convert the FHIR records again, this time applying the cryptographic shield to force exactly 30 disclosures per credential, padded :

```bash
python generate_padded_sdjwts.py
# Outputs saved to output/best_sd_jwts_padded/
Use for both Covert_both.py
```


## 5. Train and Test ML Models 

Train the 5 machine learning classifiers (Logistic Regression, SVM, KNN, Random Forest, Gradient Boosting) using the structural metadata from the **unpadded** datasets, and test their predictive accuracy directly against the **padded** datasets:

```bash
python evaluate_models.py

```
