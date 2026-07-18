import pandas as pd

def run_quality_check(filepath):
    print("--- Running Treasury Data QA Audit ---")
    df = pd.read_csv(filepath)
    
    # Check 1: Ensure Net Exposure Math is Correct
    # Net should always equal Gross - Hedge
    df['calculated_net'] = df['gross_exposure_fc'] - df['hedge_notional_fc']
    # We round to avoid small floating point errors
    math_errors = df[round(df['calculated_net'], 2) != round(df['net_exposure_fc'], 2)]
    
    if len(math_errors) == 0:
        print("[PASS] Net Exposure logic is mathematically sound.")
    else:
        print(f"[FAIL] Found {len(math_errors)} rows with bad exposure math.")

    # Check 2: Missing Data Check
    if df.isnull().values.any():
        print(f"[FAIL] Found missing values in the dataset. Check the spot rate merge.")
    else:
        print("[PASS] No null values detected.")

if __name__ == "__main__":
    # Point it to the raw file your groupmate made
    run_quality_check('data/raw/synthetic_treasury_risk_data.csv')
    
    