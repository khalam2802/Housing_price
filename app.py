import streamlit as st
import pandas as pd

st.set_page_config(page_title="🏠 Housing Price Analysis", layout="wide")
st.title("📊 Housing Price Analysis Dashboard")

st.markdown("Dữ liệu đã được phân tích từ GitHub repo `Housing_price`")

# Hiển thị dữ liệu
try:
    df = pd.read_csv("output.csv")
    st.success("✅ Dữ liệu đã tải thành công!")
    st.dataframe(df)
except Exception as e:
    st.error(f"❌ Lỗi khi tải dữ liệu: {e}")

