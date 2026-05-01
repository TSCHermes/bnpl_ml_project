import pandas as pd

# --- Configuration ---
FILE_PATH = "/opt/data/hermes_workspace/bnpl_ml_project/bnpl_dataset.csv"

def perform_eda():
    print("=========================================")
    print("         DEEPER EDA: TARGET VARIABLE ANALYSIS")
    print("=========================================")
    try:
        # 1. Load Data
        df = pd.read_csv(FILE_PATH)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")

        # 2. Target Variable Distribution
        print("\n--- Repayment Status Distribution ---")
        status_counts = df['Repayment_Status'].value_counts()
        for status, count in status_counts.items():
            print(f"{status}: {count}")

        # 3. Analyze Risk by Credit Score Bins (Simple Binning)
        print("\n--- Average Purchase Amount by Credit Score Tier ---")
        bins = [0, 500, df['Credit_Score'].max() + 1]; labels = ['Low Risk (<500)', 'Medium Risk (500-650)', 'High Risk (>650)']; 
        df['Credit_Tier'] = pd.cut(df['Credit_Score'], bins=bins, labels=labels, right=False)
        tier_avg = df.groupby('Credit_Tier')['Purchase_Amount'].mean().reset_index()
        for index, row in tier_avg.iterrows():
            print(f"{row['Credit_Tier']}: Avg Purchase Amount = {row['Purchase_Amount']:.2f}")

        # 4. Analyze Risk by Purchase Category
        print("\n--- Repayment Status Breakdown by Purchase Category ---")
        category_risk = df.groupby('Purchase_Category')['Repayment_Status'].value_counts(normalize=True).unstack(fill_value=0)
        for category in category_risk.columns:
            print(f"Category: {category}")
            for status in category_risk.index:
                if status == 'Repayment_Status': continue # Skip the index name if it appears as a column
                percentage = category_risk.loc[status, category] * 100
                print(f"  - {status}: {percentage:.2f}%")

    except FileNotFoundError:
        print(f"Error: The file was not found at {FILE_PATH}")
    except Exception as e:
        print(f"An error occurred during EDA: {e}")

if __name__ == "__main__":
    perform_eda()