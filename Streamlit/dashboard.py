import streamlit as st
import pandas as pd
from utils.database import supabase

# -------------------------
# LOAD DATA
# -------------------------
def load_attendance():
    response = supabase.table("attendance").select("*").execute()
    df = pd.DataFrame(response.data)
    return df


# -------------------------
# DASHBOARD UI
# -------------------------
def show_dashboard():

    st.subheader("📊 Attendance Dashboard")

    df = load_attendance()

    if df.empty:
        st.info("No attendance data yet")
        return

    # -------------------------
    # CLEAN DATA
    # -------------------------
    df["date"] = pd.to_datetime(df["date"])
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # -------------------------
    # DAILY ATTENDANCE CHART
    # -------------------------
    daily = df.groupby(df["date"].dt.date).size()

    st.write("### 📈 Daily Attendance")
    st.line_chart(daily)

    # -------------------------
    # CONFIDENCE ANALYSIS
    # -------------------------
    st.write("### 🎯 Confidence Distribution")
    st.bar_chart(df["confidence"])

    # -------------------------
    # TABLE VIEW
    # -------------------------
    st.write("### 📋 Attendance Records")
    st.dataframe(df.sort_values(by="date", ascending=False))