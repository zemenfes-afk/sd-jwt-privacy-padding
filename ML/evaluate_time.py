import time
from sd_jwt.issuer import SDJWTIssuer
from sd_jwt.common import SDObj
from jwcrypto import jwk

def run_time_benchmark():
    print("⏱️  EVALUATING COMPUTATIONAL TIME OVERHEAD ⏱️")
    print("="*60)
    
    # Generate a cryptographic key for the test
    issuer_key = jwk.JWK.generate(kty="EC", crv="P-256")
    
    # Simulate a Sick Patient with 3 real diseases
    standard_claims = {
        "sub": "patient-123",
        "active_conditions": [SDObj("Diabetes"), SDObj("Hypertension"), SDObj("Asthma")]
    }
    
    # Simulate the Padded Patient with 20 items (3 real + 17 dummies)
    padded_claims = {
        "sub": "patient-123",
        "active_conditions": [
            SDObj("Diabetes"), SDObj("Hypertension"), SDObj("Asthma"),
            SDObj("DUMMY_1"), SDObj("DUMMY_2"), SDObj("DUMMY_3"), SDObj("DUMMY_4"),
            SDObj("DUMMY_5"), SDObj("DUMMY_6"), SDObj("DUMMY_7"), SDObj("DUMMY_8"),
            SDObj("DUMMY_9"), SDObj("DUMMY_10"), SDObj("DUMMY_11"), SDObj("DUMMY_12"),
            SDObj("DUMMY_13"), SDObj("DUMMY_14"), SDObj("DUMMY_15"), SDObj("DUMMY_16"),
            SDObj("DUMMY_17")
        ]
    }

    ITERATIONS = 500

    # --- 1. Test Standard SD-JWT Generation Time ---
    start_time = time.perf_counter()
    for _ in range(ITERATIONS):
        SDJWTIssuer(standard_claims, issuer_key)
    standard_time = time.perf_counter() - start_time
    standard_ms_per_file = (standard_time / ITERATIONS) * 1000

    # --- 2. Test Padded SD-JWT Generation Time ---
    start_time = time.perf_counter()
    for _ in range(ITERATIONS):
        SDJWTIssuer(padded_claims, issuer_key)
    padded_time = time.perf_counter() - start_time
    padded_ms_per_file = (padded_time / ITERATIONS) * 1000

    # --- Calculate the Difference ---
    overhead_ms = padded_ms_per_file - standard_ms_per_file
    increase_percent = (padded_time - standard_time) / standard_time * 100

    print(f"🔹 Test Size: Generating {ITERATIONS} credentials per method.")
    print("-" * 60)
    print(f"1. Standard SD-JWT Time:  {standard_ms_per_file:.2f} milliseconds per file")
    print(f"2. Padded SD-JWT Time:    {padded_ms_per_file:.2f} milliseconds per file")
    print("-" * 60)
    print(f"🚨 CPU Time Overhead:     +{overhead_ms:.2f} ms per file (+{increase_percent:.2f}%)")
    print("="*60)
    
    if overhead_ms < 5.0:
        print("✅ CONCLUSION: The padding overhead is under 5 milliseconds.")
        print("   This is highly optimized and perfectly safe for real-time API use.")

if __name__ == "__main__":
    run_time_benchmark()