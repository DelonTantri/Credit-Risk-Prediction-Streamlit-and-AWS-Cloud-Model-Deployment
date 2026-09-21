import streamlit as st
from inference import CreditScoreInference

predictor = CreditScoreInference()

st.set_page_config(page_title="Credit Score Classification", layout="wide")
st.title("Credit Score Classification")
st.markdown("Cek Kelayakan Nasabah.")

# input form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0)
    bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=5)
    credit_cards = st.number_input("Number of Credit Cards", min_value=0, value=3)
    interest_rate = st.number_input("Interest Rate", min_value=0.0, value=10.0)
    num_loans = st.number_input("Number of Loans", min_value=0, value=2)
    delay_due = st.number_input("Delay From Due Date", min_value=0, value=5)
    delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, value=2)

with col2:
    changed_limit = st.number_input("Changed Credit Limit", value=5.0)
    inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=3)
    outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
    utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, value=30.0)
    credit_history_age = st.number_input("Credit History Age (Months)", min_value=0, value=120)
    payment_min = st.selectbox("Payment of Minimum Amount", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    total_emi = st.number_input("Total EMI Per Month", min_value=0.0, value=200.0)
    invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, value=500.0)
    monthly_balance = st.number_input("Monthly Balance", value=3000.0)
    
# Predict button
if st.button("Predict Credit Score", use_container_width=True):
    user_data = {
        "Age": age, "Annual_Income": annual_income, "Monthly_Inhand_Salary": monthly_salary, "Num_Bank_Accounts": bank_accounts,
        "Num_Credit_Card": credit_cards, "Interest_Rate": interest_rate, "Num_of_Loan": num_loans, "Delay_from_due_date": delay_due,
        "Num_of_Delayed_Payment": delayed_payment, "Changed_Credit_Limit": changed_limit, "Num_Credit_Inquiries": inquiries,
        "Outstanding_Debt": outstanding_debt, "Credit_Utilization_Ratio": utilization_ratio, "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": payment_min, "Total_EMI_per_month": total_emi, "Amount_invested_monthly": invested_monthly, "Monthly_Balance": monthly_balance
    }

    prediction = predictor.predict(user_data)
    st.divider()

    if prediction == "Good":
        st.success(f"Predicted Credit Score: {prediction}")
    elif prediction == "Standard":
        st.warning(f"Predicted Credit Score: {prediction}")
    else:
        st.error(f"Predicted Credit Score: {prediction}")