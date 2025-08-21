import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(page_title="Medical Billing Analyst at RCM Company.", layout="wide")
col1, col2, col3 = st.columns([1,2,1])  # middle column bigger
with col2:
        st.image(r"C:\Users\sakth\Downloads\Stempel-Primrose-SlateGreen-1024x1024.png", width=600)
st.markdown("""
           <style>
               .stApp {
                   background-color:black;  
               }
           </style>
       """, unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center;">Go ahead and upload your CSV and Excel file, and I’ll take care of</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align:center;">Summarizing trends</h3>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align:center;">Diagnosing root causes</h3>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align:center;">Recommending corrective actions to reduce denials and improve collections</h3>', unsafe_allow_html=True)
st.title("🩺 Performance Review  And Denial Management Analysis")
st.markdown(
    "Upload your billing dataset (CSV / Excel). Required columns: "
    "`CPT Code`, `Insurance Company`, `Physician Name`, `Payment Amount`, `Balance`, `Denial Reason`."
)
def read_file(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        return pd.read_excel(uploaded, engine="openpyxl")
def clean_currency(x):
    if pd.isna(x):# if the currency  is nan means it return 0
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)): # if the currency in int or float it cconvert into float
        return float(x)
    s = str(x)
    s = re.sub(r"[^\d\.\-]", "", s)  # remove $ and parentheses, etc
    if s == "" or s == "-" or s == ".":
        return 0.0
    try:
        return float(s)
    except:
        return 0.0
def normalize_columns(df):
    # Map common column name variants to canonical names
    # why mapping these are the important columns for medical billing
    mapping = {}
    cols = list(df.columns)
    for c in cols:
        cl = c.strip().lower() # " CPT code" into "Cpt code"
        if "cpt" in cl and "code" in cl:
            mapping[c] = "CPT Code"
        elif cl in ("cpt", "cptcode"):
            mapping[c] = "CPT Code"
        elif "insurance" in cl or "payer" in cl:
            mapping[c] = "Insurance Company"
        elif "physician" in cl or "provider" in cl or "doctor" in cl:
            mapping[c] = "Physician Name"
        elif "payment" in cl and "amount" in cl or cl=="payment":
            mapping[c] = "Payment Amount"
        elif "balance" in cl or "outstanding" in cl:
            mapping[c] = "Balance"
        elif "denial" in cl:
            mapping[c] = "Denial Reason"
    df = df.rename(columns=mapping)
    for required in ["CPT Code", "Insurance Company", "Physician Name", "Payment Amount", "Balance", "Denial Reason"]:
        if required not in df.columns:
            df[required] = pd.NA
    return df
def prepare_df(df):
    df = df.copy()
    df = df.dropna(how="all")  # drop full empty rows
    df = normalize_columns(df)
    df["CPT Code"] = df["CPT Code"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["Payment Amount"] = df["Payment Amount"].apply(clean_currency)
    df["Balance"] = df["Balance"].apply(clean_currency)
    df["Denial Reason"] = df["Denial Reason"].astype(str).replace({"nan": None})
    df["Is Denied"] = df["Denial Reason"].notna() & (df["Denial Reason"].str.strip() != "")
    df["Is Zero Payment"] = df["Payment Amount"] == 0
    return df
def top_denied_cpts(df, n=20):
    t = df.groupby("CPT Code", dropna=False).agg(
        total_claims=("CPT Code", "count"),
        denied_claims=("Is Denied", "sum"),
        total_balance=("Balance","sum"),
        total_paid=("Payment Amount","sum")
    ).reset_index()
    t["Denial Rate"] = (t["denied_claims"] / t["total_claims"]).fillna(0)
    return t.sort_values("denied_claims", ascending=False).head(n)
def top_paid_cpts(df, n=20):
    t = df.groupby("CPT Code").agg(total_paid=("Payment Amount","sum"), total_claims=("CPT Code","count")).reset_index()
    return t.sort_values("total_paid", ascending=False).head(n)
def denials_by_payer(df):
    t = df.groupby("Insurance Company").agg(
        total_claims=("CPT Code","count"),
        denied_claims=("Is Denied","sum"),
        total_paid=("Payment Amount","sum"),
        total_balance=("Balance","sum")
    ).reset_index()
    t["Denial Rate"] = t["denied_claims"]/t["total_claims"]
    return t.sort_values("denied_claims", ascending=False)
def denials_by_provider(df):
    t = df.groupby("Physician Name").agg(
        total_claims=("CPT Code","count"),
        denied_claims=("Is Denied","sum"),
        total_paid=("Payment Amount","sum"),
        total_balance=("Balance","sum")
    ).reset_index()
    t["Denial Rate"] = t["denied_claims"]/t["total_claims"]
    return t.sort_values("denied_claims", ascending=False)
def denial_reason_summary(df):
    t = df[df["Is Denied"]]["Denial Reason"].value_counts().reset_index()
    t.columns = ["Denial Reason", "Count"]
    return t
ROOT_CAUSE_MAP = {
    "16": ("Missing information", "Ensure patient demographic/insurance fields & documentation; resubmit."),
    "45": ("Charge exceeds fee schedule", "Check contract/fee schedule, validate coding and consider appeal."),
    "96": ("Non-covered service", "Confirm benefit coverage; obtain prior auth or collect patient responsibility."),
    "197": ("Prior auth required", "Obtain/submit prior authorization and supporting documentation."),
    "109": ("Provider not credentialed", "Verify provider enrollment with payer and correct NPI/credentials."),
}
def detect_root_causes(df):
    # root causes detection  and recommended fixes
    detected = []
    reasons = df["Denial Reason"].dropna().unique()
    for r in reasons:
        m = re.match(r"\s*(\d{1,4})", str(r))
        if m:
            code = m.group(1)
            if code in ROOT_CAUSE_MAP:
                detected.append((r, code, ROOT_CAUSE_MAP[code][0], ROOT_CAUSE_MAP[code][1]))
    return detected
def appeal_opportunities(df):
    fixable_codes = set(ROOT_CAUSE_MAP.keys())
    rows = []
    for idx, row in df[df["Is Denied"]].iterrows():
        m = re.match(r"\s*(\d{1,4})", str(row["Denial Reason"]))
        if m and m.group(1) in fixable_codes:
            rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(columns=df.columns)
def cpt_payer_heatmap(df):
    pivot = df[df["Is Denied"]].pivot_table(index="CPT Code", columns="Insurance Company", values="Is Denied", aggfunc="sum", fill_value=0)
    return pivot
# UI Starts   Here
st.image(r"C:\Users\sakth\Downloads\AI-in-Medical-Billing.jpg")
uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
st.snow()
# st.markdown("""
#            <style>
#                .stApp {
#                    background-color:#2E8B57;
#                }
#            </style>
#        """, unsafe_allow_html=True)
if uploaded is None:
    st.info("Upload your billing CSV or Excel file to begin. (Example columns shown in the top text.)")
    st.stop()
try:
    df_raw = read_file(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()
df = prepare_df(df_raw)
st.sidebar.header("Summary Metrics")

st.markdown(
    """
    <style>
        /* Change Sidebar background color */
        section[data-testid="stSidebar"] {
            background-color: #D3D3D3; ;
        }

        /* Optional: change sidebar text color */
        section[data-testid="stSidebar"] * {
            color: black;  /* you can also use white if bg is dark */
        }
    </style>
    """,
    unsafe_allow_html=True
)
total_claims = len(df)
total_denied = int(df["Is Denied"].sum())
total_paid = df["Payment Amount"].sum()
total_balance = df["Balance"].sum()
denied_balance = df[df["Is Denied"]]["Balance"].sum()
st.sidebar.metric("Total Claims", f"{total_claims}")
st.sidebar.metric("Total Denied", f"{total_denied}")
st.sidebar.metric("Total Paid", f"${total_paid:,.2f}")
st.sidebar.metric("Total Balance", f"${total_balance:,.2f}")
st.sidebar.metric("Denied Balance", f"${denied_balance:,.2f}")
# tabs for 19 questions :
tabs = st.tabs([
    "CPT Analysis", "Payer Analysis", "Provider Analysis",
    "Denial Reasons", "Root Causes & Fixes", "Revenue Impact",
    "Zero Payments & High Balance", "Appeal Opportunities", "Visuals & Heatmap", "Export Reports"
])
# Tab 1: CPT Analysis (1,2,3,9)
with tabs[0]:
    st.balloons()
    st.header("CPT Analysis")
    st.markdown("Top denied CPTs, denial rates, and top paid CPTs.")
    top_denied = top_denied_cpts(df, n=50)
    st.subheader("Top Denied CPT Codes (by count)")
    st.dataframe(top_denied)

    st.subheader("Denial Rate per CPT")
    cpt_rate = top_denied[["CPT Code", "total_claims", "denied_claims", "Denial Rate"]].sort_values("Denial Rate", ascending=False)
    st.dataframe(cpt_rate)

    st.subheader("Top Paid CPT Codes")
    st.dataframe(top_paid_cpts(df, n=50))

    st.subheader("Revenue Lost by CPT (Outstanding Balance on Denied Claims)")
    loss_by_cpt = df[df["Is Denied"]].groupby("CPT Code").agg(lost_balance=("Balance","sum"), denied_count=("Is Denied","sum")).reset_index().sort_values("lost_balance", ascending=False)
    st.dataframe(loss_by_cpt)

# Tab 2: Payer Analysis (4,7,10,12)
with tabs[1]:
    st.header("Payer Analysis")
    payer_tbl = denials_by_payer(df)
    st.subheader("Denials by Insurance Company")
    st.dataframe(payer_tbl)

    st.subheader("Denial Reasons by Payer (top reasons)")
    dr_payer = df[df["Is Denied"]].groupby(["Insurance Company","Denial Reason"]).size().reset_index(name="Count").sort_values(["Insurance Company","Count"], ascending=[True,False])
    st.dataframe(dr_payer)

    st.subheader("Revenue Lost by Payer")
    loss_by_payer = df[df["Is Denied"]].groupby("Insurance Company").agg(lost_balance=("Balance","sum"), denied_count=("Is Denied","sum")).reset_index().sort_values("lost_balance", ascending=False)
    st.dataframe(loss_by_payer)

    st.subheader("Payer behavior — top CPTs denied per payer")
    payer_cpt = df[df["Is Denied"]].groupby(["Insurance Company","CPT Code"]).size().reset_index(name="Count").sort_values(["Insurance Company","Count"], ascending=[True,False])
    st.dataframe(payer_cpt)

# Tab 3: Provider Analysis (5,8,11,13)
with tabs[2]:

    st.header("Provider / Physician Analysis")
    prov_tbl = denials_by_provider(df)
    st.subheader("Denials by Physician")
    st.dataframe(prov_tbl)

    st.subheader("Denial Reasons by Physician")
    dr_prov = df[df["Is Denied"]].groupby(["Physician Name","Denial Reason"]).size().reset_index(name="Count").sort_values(["Physician Name","Count"], ascending=[True,False])
    st.dataframe(dr_prov)

    st.subheader("Revenue Lost by Physician")
    loss_by_prov = df[df["Is Denied"]].groupby("Physician Name").agg(lost_balance=("Balance","sum"), denied_count=("Is Denied","sum")).reset_index().sort_values("lost_balance", ascending=False)
    st.dataframe(loss_by_prov)

    st.subheader("Provider behavior trends — which providers get which denial codes most")
    prov_code = df[df["Is Denied"]].groupby(["Physician Name"]).apply(lambda d: d["Denial Reason"].value_counts().head(5)).reset_index()
    prov_code.columns = ["Physician Name", "Denial Reason", "Count"]
    st.dataframe(prov_code)

# Tab 4: Denial Reasons (6)
with tabs[3]:
    st.header("Denial Reasons Summary")
    reasons = denial_reason_summary(df)
    st.dataframe(reasons)
    fig = px.bar(reasons, x="Denial Reason", y="Count", title="Denial Reasons")
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Root Causes & Fixes (14,15)
with tabs[4]:
    st.header("Root Cause Detection & Recommended Fixes")
    detected = detect_root_causes(df)
    if detected:
        rows = []
        for full_text, code, cause, fix in detected:
            rows.append({"Denial Reason": full_text, "Code": code, "Likely Cause": cause, "Recommended Fix": fix})
        rc_df = pd.DataFrame(rows)
        st.table(rc_df)
    else:
        st.success("No mapped root causes found in denial reasons (based on code map).")

    st.markdown("**Generic flags & checks** the app applies:")
    st.write("- Modifier issues: Check for common modifier keywords in Denial Reason or custom flags in your system.")
    st.write("- LCD/NCD mismatch: Denials mentioning 'not covered under LCD/NCD' should be reviewed by clinical team.")
    st.write("- Bundling (NCCI): Denials referencing fee schedule or bundling should be checked for NCCI edits.")
    st.write("- Documentation: Missing information denials usually require provider notes and demographic validation.")
    st.write("- Prior authorization: Denials with 'prior auth' codes require fetching prior auth documentation.")
    st.write("- Credentialing/enrollment: 'Provider not credentialed' denials require payer enrollment check.")

# Tab 6: Revenue Impact (9,10,11)
with tabs[5]:
    st.header("Revenue Impact")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Paid", f"${total_paid:,.2f}")
    col2.metric("Total Outstanding Balance", f"${total_balance:,.2f}")
    col3.metric("Outstanding on Denied Claims", f"${denied_balance:,.2f}")

    st.subheader("Top CPTs by Outstanding Denied Balance")
    st.dataframe(loss_by_cpt := df[df["Is Denied"]].groupby("CPT Code").agg(Outstanding=("Balance","sum"), Denied_Count=("Is Denied","sum")).reset_index().sort_values("Outstanding", ascending=False))

    st.subheader("Top Payers by Outstanding Denied Balance")
    st.dataframe(loss_by_payer) # payer analysis we create function

    st.subheader("Top Physicians by Outstanding Denied Balance")
    st.dataframe(loss_by_prov) #provider  analysis we create function

# Tab 7: Zero Payments & High Balance (16,17)
with tabs[6]:
    # patient has high balances and insurance company not paid
    st.header("Zero Payments & High Balances")
    st.subheader("Patient Has High Balances And Insurance Company Not Paid ")
    zero_pay = df[df["Is Zero Payment"]]
    st.subheader(f"Zero Payment Claims (count: {len(zero_pay)})")
    st.dataframe(zero_pay)

    st.subheader("High Balance Accounts (top 50)")
    high_bal = df.sort_values("Balance", ascending=False).head(50)
    st.dataframe(high_bal)

# Tab 8: Appeal Opportunities (18)
with tabs[7]:
    st.header("Appeal Opportunities (Fixable Denials)")
    appeals = appeal_opportunities(df)
    st.write("Criteria: denials whose code is in known fixable list (e.g., 16,45,96,197,109).")
    if not appeals.empty:
        st.dataframe(appeals)
        st.markdown("**Suggested next steps for appeals:**")
        st.write("- Gather supporting documentation (notes, labs, imaging).")
        st.write("- Correct demographics/insurance data and resubmit if missing info.")
        st.write("- If fee schedule issue, validate coding and contract; consider corrected claim or appeal.")
        st.write("- For non-covered/authorization issues, confirm coverage or collect patient responsibility.")
    else:
        st.success("No immediate appeal opportunities detected by the simple rule.")

# Tab 9: Visuals & Heatmap (19)
with tabs[8]:
    st.header("Visual Reporting")
    st.subheader("Bar charts (interactive)")
    # small charts
    c1, c2 = st.columns(2)
    figA = px.bar(top_denied.head(10), x="CPT Code", y="denied_claims", title="Top 10 Denied CPTs")
    c1.plotly_chart(figA, use_container_width=True)
    figB = px.bar(payer_tbl.head(10), x="Insurance Company", y="denied_claims", title="Top 10 Payers by Denials")
    c2.plotly_chart(figB, use_container_width=True)

    st.subheader("Heatmap: Denied counts by CPT (rows) × Payer (columns)")
    heat = cpt_payer_heatmap(df)
    if heat.empty:
        st.info("No denied rows to create heatmap.")
    else:
        heat_small = heat.copy()
        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_small.values,#number od denials
            x=heat_small.columns.tolist(),#payers
            y=heat_small.index.tolist(),#cpt codes
            colorscale="Reds",
            hoverongaps=False
        ))
        fig_heat.update_layout(height=600, title="Denied Claims Heatmap (CPT × Payer)")
        st.plotly_chart(fig_heat, use_container_width=True)
import zipfile
with tabs[9]:
    st.header("Export / Download Reports")
    st.write("Download cleaned dataset, summary sheets, and appeal list as CSV files (zipped).")
    to_export = {
        "cleaned": df,
        "top_denied_cpts": top_denied_cpts(df, n=200),
        "top_paid_cpts": top_paid_cpts(df, n=200),
        "denials_by_payer": denials_by_payer(df),
        "denials_by_provider": denials_by_provider(df),
        "denial_reasons": denial_reason_summary(df),
        "appeal_opportunities": appeal_opportunities(df)
    }
    buffer = io.BytesIO() # creates an in-memory file (like a virtual file stored in RAM).
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in to_export.items():# Loops over each report we want to export.
            #name = "top_denied_cpts", data = top_denied_cpts(df, n=200)
            csv_bytes = data.to_csv(index=False).encode("utf-8")
             # Converts each DataFrame into a CSV string using .to_csv().
            # .encode("utf-8") turns it into bytes, which are required for writing into a zip file.
            zf.writestr(f"{name}.csv", csv_bytes)
            # f"{name}.csv" = filename inside the zip (example: cleaned.csv, denials_by_payer.csv).
            # csv_bytes = actual CSV content.
    buffer.seek(0)
    # Moves the file pointer back to the start of the buffer.
    st.download_button(
        label="⬇️ Download All Reports (ZIP of CSVs)",
        data=buffer, # gives Streamlit the in-memory zip file.
        file_name="denial_reports.zip", # name of denial Zips
        mime="application/zip" # tells the browser it's a zip file.
    )


