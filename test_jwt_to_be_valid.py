import base64
import json

def decode_base64url(b64_string):
    """
    Standard Base64 decoders crash if the string isn't perfectly divisible by 4.
    This helper function adds the missing '=' padding required for Base64Url.
    """
    padded = b64_string + '=' * (4 - len(b64_string) % 4)
    return base64.urlsafe_b64decode(padded).decode('utf-8')

def decode_sd_jwt(sd_jwt_string):
    print("🔍 DECODING SD-JWT...\n" + "="*60)
    
    # STEP 1: Split the main JWT from the hidden Disclosures using '~'
    parts = sd_jwt_string.split('~')
    jwt_token = parts[0]
    disclosures = parts[1:]

    # STEP 2: Decode the JWT (Split by '.')
    jwt_parts = jwt_token.split('.')
    if len(jwt_parts) >= 2:
        try:
            header = json.loads(decode_base64url(jwt_parts[0]))
            payload = json.loads(decode_base64url(jwt_parts[1]))
            
            print("\n1️⃣  JWT HEADER:")
            print(json.dumps(header, indent=2))
            
            print("\n2️⃣  JWT PAYLOAD (The visible data + the hashes):")
            print(json.dumps(payload, indent=2))
        except Exception as e:
            print(f"❌ Error decoding JWT: {e}")

    # STEP 3: Decode the Disclosures (The hidden data)
    print("\n3️⃣  DECODED DISCLOSURES (The hidden plaintext values):")
    for i, disclosure in enumerate(disclosures):
        if not disclosure: # Ignore the empty string at the very end
            continue
            
        try:
            # Disclosures are JSON arrays: ["Salt", "Plaintext Value"]
            decoded_disclosure = json.loads(decode_base64url(disclosure))
            print(f"  [+] Disclosure {i+1}: {decoded_disclosure}")
        except Exception as e:
            print(f"  [?] Item {i+1} (Could be a Key Binding signature): {disclosure[:20]}...")
            
    print("="*60)

# ==========================================
# PASTE YOUR SD-JWT STRING HERE TO TEST IT
# ==========================================
filename="output/sd_jwts/Aaron697_Moore224_1f2624da-6e90-5626-a13e-019a45909965.txt"
with open(filename, 'r') as fl:
    sample_string = fl.read().strip()


if __name__ == "__main__":
    decode_sd_jwt(sample_string)