import json
import os
from sd_jwt.issuer import SDJWTIssuer
from sd_jwt.common import SDObj  # <--- IMPORT THIS
import random
import string
import tqdm
import base64
# Try to import the demo helper; if unavailable, fall back to generating
# a jwcrypto key so the script runs for local testing.


import os
from jwcrypto import jwk

padding=True

KEY_FILE = "private_key.json"

def get_default_key():
    # 1. Check if the key already exists on disk
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            # Load the existing key from the file
            return jwk.JWK.from_json(f.read())
    
    # 2. If not, generate a new one
    print("No key found. Generating a new P-256 EC key...")
    key = jwk.JWK.generate(kty="EC", crv="P-256")
    
    # 3. Save the private key to disk for next time
    # 'export_private=True' ensures we save the full key, not just the public part
    with open(KEY_FILE, "w") as f:
        f.write(key.export_private())
        
    return key

# Create the issuer key (demo only)
issuer_key = get_default_key()

# --- CONFIGURATION ---
INPUT_DIR = "output/fhir/"       # Path to your Synthea JSON files
OUTPUT_DIR_PADDED = "output/final_sd_jwt_padded/"
OUTPUT_DISCLOSURE_DIR_PADDED = "output/final_sd_jwt_padded_disclosure/"   # Path to save the SD-JWTs
os.makedirs(OUTPUT_DIR_PADDED, exist_ok=True)
os.makedirs(OUTPUT_DISCLOSURE_DIR_PADDED, exist_ok=True)
OUTPUT_DIR_UNPADDED = "output/final_sd_jwt_unpadded/"
OUTPUT_DISCLOSURE_DIR_UNPADDED = "output/final_sd_jwt_unpadded_disclosure/"   # Path to save the SD-JWTs
os.makedirs(OUTPUT_DIR_UNPADDED, exist_ok=True)
os.makedirs(OUTPUT_DISCLOSURE_DIR_UNPADDED, exist_ok=True)



TARGET_CONDITIONS = 30

def convert_patient(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
    except Exception:
        return None

    # Find Patient resource and Conditions
    patient_id = None
    given = ""
    family = ""
    conditions = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")

        if rtype == "Patient" and not patient_id:
            patient_id = resource.get("id", "unknown-sub")
            try:
                name = resource.get("name", [])[0]
                given = " ".join(name.get("given", [])) if name.get("given") else ""
                family = name.get("family", "") if name.get("family") else ""
            except Exception:
                pass

        if rtype == "Condition":
            clinical = resource.get("clinicalStatus", {})
            coding = clinical.get("coding", [])
            code = coding[0].get("code") if coding else None
            if (code and code.lower() == "active") or not code:
                text = resource.get("code", {}).get("text")
                if text and 'disorder' in text.lower():
                    conditions.append(text)
                    #print(text)

    if not patient_id:
        return None
    
    if padding:
    # Pad the conditions list to ensure it always has TARGET_CONDITIONS items
        needed_dummies = TARGET_CONDITIONS - len(conditions)
        for _ in range(max(0, needed_dummies)):
            conditions.append("DUMMY_" + ''.join(random.choices(string.ascii_letters, k=10)))
        conditions = conditions[:TARGET_CONDITIONS]

    # --- THE FIX IS HERE ---
    # We explicitly wrap the data we want to hide in 'SDObj'.
    # This tells the library: "Hide these individual items"
    
    # Option 1: Hide the items in the list (The attacker sees a list of hashes)
    hidden_conditions = [SDObj(c) for c in conditions]
    
    # Option 2: Hide the whole list (The attacker sees one hash for the whole array)
    #hidden_conditions = SDObj(conditions) 
    #print("Hidden Conditions:", hidden_conditions)

    user_claims = {
        "iss": "https://my-hospital.example",
        "sub": patient_id,
        "given_name": given or "Unknown",
        "family_name": family or "Unknown",
        "active_conditions": hidden_conditions,  # Pass the wrapped data
    }

    
    # Now we just pass the claims. The library sees the SDObj and hides it automatically.
    issuer = SDJWTIssuer(user_claims, issuer_key)
    
    sdissued=issuer.sd_jwt_issuance
    #print("Generated SD-JWT:", sdissued)
    jwt=sdissued[:sdissued.find("~")]
    disclosure=sdissued[sdissued.find("~"):]

    disclosure_arr=disclosure.split("~")
    disc=""
    for d in disclosure_arr:
        try:
            copy=d
            while len(copy) % 4 != 0:
                copy += "="
            decoded = base64.urlsafe_b64decode(copy).decode('utf-8', errors='ignore')
            if 'dummy' not in decoded.lower():
                disc=disc+decoded+"~"
        except Exception as e:
            print("Error decoding disclosure:", e)
            pass
        
    return jwt, disc


def process(OUTPUT_DIR, OUTPUT_DISCLOSURE_DIR):
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory '{INPUT_DIR}' does not exist. Please run Synthea first.")
        raise SystemExit(1)

    print(f"Converting files from {INPUT_DIR} ...")
    count = 0

    for filename in tqdm.tqdm(os.listdir(INPUT_DIR)):
        if not filename.endswith(".json"):
            continue
        in_path = os.path.join(INPUT_DIR, filename)
        try:
            credential, disclosure = convert_patient(in_path)
            if credential:
                out_name = filename.rsplit(".json", 1)[0] + ".txt"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                with open(out_path, "w", encoding="utf-8") as out:
                    out.write(credential)
                disclosure_out_path = os.path.join(OUTPUT_DISCLOSURE_DIR, out_name)
                with open(disclosure_out_path, "w", encoding="utf-8") as out:
                    out.write(disclosure)
                count += 1
                #if count % 100 == 0:
                #    print(f"Processed {count} patients...")
        except Exception as e:
            # print(f"Error processing {filename}: {e}")
            continue

    print(f"Done! Generated {count} SD-JWTs in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    padding=False
    print("Processing unpadded")
    process(OUTPUT_DIR_UNPADDED, OUTPUT_DISCLOSURE_DIR_UNPADDED)
    padding=True
    print("Processing padded")
    process(OUTPUT_DIR_PADDED, OUTPUT_DISCLOSURE_DIR_PADDED)
    