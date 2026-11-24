import io
import os
import pandas as pd
import numpy as np


def load_dataset(uploaded_file):
    """
    Safely load CSV or Excel file from Streamlit file uploader or file path.
    Supports .csv, .xlsx, .xls
    """
    if uploaded_file is None:
        return None

    filename = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except (UnicodeDecodeError, UnicodeError):
                if hasattr(uploaded_file, "seek"):
                    uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
        else:
            # Attempt CSV first, then Excel
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                if hasattr(uploaded_file, "seek"):
                    uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)

        if df is None or df.empty:
            raise ValueError("The uploaded file contains no data or could not be parsed.")
        if len(df.columns) < 2:
            raise ValueError("The dataset must have at least 2 columns (features and a target).")
        return df
    except Exception as e:
        raise ValueError(f"Failed to read file '{filename}': {str(e)}")


def load_csv(uploaded_file):
    """Backward compatibility wrapper."""
    return load_dataset(uploaded_file)


def generate_classification_demo(n=400):
    """
    Generates a realistic Customer Churn dataset with mixed numeric and categorical features.
    """
    np.random.seed(42)
    contract_types = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internet_services = ["Fiber optic", "DSL", "No"]
    
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20.0, 115.0, n), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 50, n), 2)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    contract = np.random.choice(contract_types, n, p=[0.55, 0.25, 0.20])
    payment = np.random.choice(payment_methods, n)
    internet = np.random.choice(internet_services, n, p=[0.45, 0.35, 0.20])
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    support_tickets = np.random.poisson(1.2, n)

    # Churn probability based on realistic business logic
    score = (
        (contract == "Month-to-month") * 1.5
        + (monthly_charges > 70) * 1.0
        + (support_tickets > 2) * 1.2
        - (tenure > 36) * 1.5
        - (contract == "Two year") * 1.8
        + (internet == "Fiber optic") * 0.6
        + np.random.normal(0, 0.8, n)
    )
    churn = (score > 0.8).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "payment_method": payment,
        "internet_service": internet,
        "senior_citizen": senior_citizen,
        "support_tickets": support_tickets,
        "churn": churn
    })
    return df


def generate_regression_demo(n=400):
    """
    Generates a realistic Real Estate Housing Price dataset with mixed numeric and categorical features.
    """
    np.random.seed(42)
    locations = ["Urban Center", "Suburban North", "Suburban South", "Rural West", "Coastal Metro"]
    property_types = ["Single Family", "Condo", "Townhouse", "Multi-Family"]
    condition_ratings = ["Fair", "Good", "Excellent", "Luxury"]
    
    sqft = np.random.randint(700, 4500, n)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.round(bedrooms * 0.75 + np.random.choice([0, 0.5, 1.0], n), 1)
    lot_size = np.random.randint(1500, 12000, n)
    building_age = np.random.randint(1, 65, n)
    
    location = np.random.choice(locations, n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    prop_type = np.random.choice(property_types, n)
    condition = np.random.choice(condition_ratings, n, p=[0.2, 0.45, 0.25, 0.1])
    garage_spaces = np.random.choice([0, 1, 2, 3], n, p=[0.15, 0.35, 0.4, 0.1])

    loc_multiplier = {
        "Urban Center": 1.35,
        "Coastal Metro": 1.5,
        "Suburban North": 1.1,
        "Suburban South": 1.0,
        "Rural West": 0.8
    }
    loc_factor = np.array([loc_multiplier[loc] for loc in location])
    cond_factor = np.array([{"Fair": 0.85, "Good": 1.0, "Excellent": 1.15, "Luxury": 1.35}[c] for c in condition])
    
    base_price = (
        sqft * 185
        + bedrooms * 18000
        + bathrooms * 14000
        + garage_spaces * 12000
        - building_age * 1200
    ) * loc_factor * cond_factor + np.random.normal(0, 25000, n)

    price = np.round(np.maximum(base_price, 85000), -2).astype(int)

    df = pd.DataFrame({
        "sqft_living": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "lot_size": lot_size,
        "building_age": building_age,
        "garage_spaces": garage_spaces,
        "location": location,
        "property_type": prop_type,
        "condition": condition,
        "price_usd": price
    })
    return df


def results_to_dataframe(results):
    """Convert training results dictionary to a sorted pandas DataFrame."""
    rows = []
    for name, info in results.items():
        if info.get("error"):
            row = {"Model": name, "Training Time (s)": 0, "Status": "Failed"}
        else:
            row = {"Model": name, "Training Time (s)": info.get("training_time", 0), "Status": "Success"}
            row.update(info.get("metrics", {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def save_sample_datasets():
    """Generates and writes CSV and Excel files to sample_data/ for user convenience."""
    os.makedirs("sample_data", exist_ok=True)
    
    # Classification sample
    clf_df = generate_classification_demo(n=350)
    clf_df.to_csv("sample_data/customer_churn_classification.csv", index=False)
    clf_df.to_excel("sample_data/customer_churn_classification.xlsx", index=False)
    
    # Regression sample
    reg_df = generate_regression_demo(n=350)
    reg_df.to_csv("sample_data/real_estate_regression.csv", index=False)
    reg_df.to_excel("sample_data/real_estate_regression.xlsx", index=False)
