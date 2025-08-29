# ui/guides.py
import streamlit as st

def render_guides_tab():
    """Renders the guides tab with financial information."""
    st.subheader("Financial Guides")
    st.markdown("General information to help you on your financial journey. For personalized advice, please consult a professional.")

    tax_expander = st.expander("🇮🇳 A Brief Guide to Indian Income Tax")
    with tax_expander:
        st.markdown("""
        #### Understanding Tax Slabs
        India has a progressive tax system with different tax slabs. As of the latest updates, there are two tax regimes: the **Old Regime** and the **New Regime**.

        - **Old Regime:** Allows you to claim various deductions and exemptions like HRA, Section 80C (for investments in PF, ELSS, etc.), 80D (medical insurance), and more.
        - **New Regime:** Offers lower tax rates but you must forgo most of the major deductions and exemptions.

        It's crucial to calculate your tax liability under both regimes to see which is more beneficial for you.

        #### Key Deductions (under Old Regime)
        - **Section 80C:** Up to ₹1.5 lakh deduction for investments in Provident Fund (PF), Public Provident Fund (PPF), Equity Linked Savings Scheme (ELSS), life insurance premiums, etc.
        - **Section 80D:** Deduction for health insurance premiums paid for self, family, and parents.
        - **House Rent Allowance (HRA):** If you live in a rented house, you can claim HRA exemption.

        > **Disclaimer:** Tax laws are subject to change. This is for informational purposes only. Always consult a certified tax professional for official advice.
        """)

    investment_expander = st.expander("💡 Introduction to Savings & Investments")
    with investment_expander:
        st.markdown("""
        #### The 50/30/20 Budgeting Rule
        A simple and effective way to manage your money:
        - **50% on Needs:** Essentials like rent/EMI, groceries, utilities, and transport.
        - **30% on Wants:** Lifestyle expenses like dining out, shopping, entertainment.
        - **20% on Savings & Investments:** This is the crucial part for your future. Aim to save/invest at least 20% of your income.

        #### Where to Invest?
        - **Low Risk:** Public Provident Fund (PPF), Fixed Deposits (FDs), Debt Mutual Funds. Good for capital preservation.
        - **Medium Risk:** Balanced or Hybrid Mutual Funds (mix of equity and debt).
        - **High Risk:** Equity Mutual Funds, Direct Stocks. Offer potential for higher returns but come with higher risk.

        **Systematic Investment Plan (SIP):** A SIP is a great way to start investing in mutual funds. It allows you to invest a fixed amount regularly (e.g., monthly), which helps in averaging out your purchase cost and builds discipline.

        > **Disclaimer:** Investing involves market risk. This information is for educational purposes. Consult a financial advisor before making investment decisions.
        """)