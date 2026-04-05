import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------------------------------
# Load data and model
# -------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Hiya\07-Coding\MachineLearning\RetailForecast\Dataset\train.csv")

@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\DELL\OneDrive\Documents\Hiya\07-Coding\MachineLearning\RetailForecast\model.pkl")

train = load_data()
model = load_model()


# -------------------------------------------------------
# Prediction function for a given date
# -------------------------------------------------------
def predict_sales_for_date(date_str, model, train):
    date = pd.to_datetime(date_str)

    # Unique stores
    stores = train["Store"].unique()

    # Create a record for each store
    input_df = pd.DataFrame({
        "Store": stores,
        "Year": date.year,
        "Month": date.month,
        "Day": date.day
    })

    # Add any other expected columns dynamically
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            if col in train.columns:
                # Fill with mode (most frequent) value
                input_df[col] = train[col].mode()[0]
            else:
                input_df[col] = 0  # fallback for missing cols

    # Ensure correct feature order
    X_input = input_df[model.feature_names_in_]

    # Predict sales per store
    predictions = model.predict(X_input)
    input_df["Predicted_Sales"] = predictions

    # Sort and get top 10 stores
    top_stores = input_df.sort_values("Predicted_Sales", ascending=False).head(10)

    # Compute overall summary
    total_predicted_sales = input_df["Predicted_Sales"].sum()
    avg_sales = train["Sales"].mean()

    summary = (
        f"📅 For **{date.strftime('%B %d, %Y')}**, "
        f"the model predicts total sales across all stores to be around **{total_predicted_sales:,.0f} units**.\n\n"
        f"The top-performing store is **Store {top_stores.iloc[0]['Store']}**, "
        f"with expected sales of approximately **{top_stores.iloc[0]['Predicted_Sales']:,.0f} units**.\n\n"
        f"This day’s performance is expected to be "
        f"{'stronger' if total_predicted_sales > avg_sales else 'slightly below average'} "
        f"compared to the usual sales trend."
    )

    # Visualization: top 10 stores
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(top_stores["Store"].astype(str), top_stores["Predicted_Sales"], color="royalblue")
    ax.set_title(f"Top 10 Stores by Predicted Sales on {date.strftime('%b %d, %Y')}")
    ax.set_xlabel("Store ID")
    ax.set_ylabel("Predicted Sales")
    plt.xticks(rotation=45)

    return summary, top_stores, total_predicted_sales, fig


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Retail Forecast Dashboard", layout="centered")
st.title("🛍️ Retail Sales Forecasting Dashboard")

st.markdown(
    """
    Predict and visualize **store-wise retail sales** for a specific date using a trained machine learning model.  
    Enter a date below to forecast how your stores are expected to perform.
    """
)

# Date input
date_input = st.date_input("📆 Select a date for prediction:", datetime.now())

if st.button("🔮 Predict Sales"):
    summary, top_stores, total_predicted_sales, fig = predict_sales_for_date(str(date_input), model, train)
    
    # Plot
    st.pyplot(fig)

    # Summary paragraph
    st.markdown("### 🧾 Prediction Summary")
    st.markdown(summary)

    # Show top 10 table
    st.markdown("### 🏆 Top 10 Performing Stores")
    st.dataframe(
        top_stores[["Store", "Predicted_Sales"]]
        .rename(columns={"Store": "Store ID", "Predicted_Sales": "Predicted Sales"})
        .style.format({"Predicted Sales": "{:,.0f}"})
    )

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, pandas, and scikit-learn.")
