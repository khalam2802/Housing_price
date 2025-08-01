import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pymongo import MongoClient

# Cấu hình giao diện
st.set_page_config(layout="wide", page_title="Phân tích & Dự đoán Giá Nhà")
st.title("🏠 Ứng dụng Phân tích & Dự đoán Giá Nhà")

# Kết nối MongoDB
client = MongoClient("mongodb+srv://khalam28:khalam28@cluster0.je2xxuc.mongodb.net")
db = client["housing_price"]
collection = db["data"]

# Tải dữ liệu từ MongoDB
data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

# Load mô hình
import zipfile
import os

# Giải nén file model
if not os.path.exists("house_price_rf_model.pkl"):
    with zipfile.ZipFile("house_price_rf_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")

model = joblib.load("house_price_rf_model.pkl")


# Tabs giao diện
menu = ["📊 Thống kê & Biểu đồ", "📂 Dữ liệu", "🤖 Dự đoán Giá Nhà"]
choice = st.sidebar.radio("📌 Chọn chức năng:", menu)

if choice == "📂 Dữ liệu":
    st.subheader("📂 Dữ liệu từ MongoDB")
    st.dataframe(df.head(200), use_container_width=True)

elif choice == "📊 Thống kê & Biểu đồ":
    st.subheader("📊 Tổng quan dữ liệu")

    col1, col2, col3 = st.columns(3)
    col1.metric("🏘️ Số lượng nhà", df.shape[0])
    col2.metric("💲 Giá trung bình", f"{df['price'].mean():,.0f} USD")
    col3.metric("📐 Diện tích TB", f"{df['sqft_living'].mean():,.0f} sqft")

    st.markdown("---")

    # Biểu đồ phân phối giá nhà
    st.markdown("### 📈 Phân phối Giá Nhà")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["price"], kde=True, color="steelblue", bins=40, ax=ax)
    ax.set_title("Biểu đồ phân phối giá nhà", fontsize=14)
    ax.set_xlabel("Giá nhà (USD)")
    st.pyplot(fig)

    # Biểu đồ hộp giá theo số phòng ngủ
    st.markdown("### 📊 Hộp Số Giá Theo Số Phòng Ngủ")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="bedrooms", y="price", data=df, palette="viridis", ax=ax)
    ax.set_title("Giá nhà theo số phòng ngủ")
    st.pyplot(fig)

    # Biểu đồ scatter diện tích - giá
    st.markdown("### 📌 Quan hệ Diện tích và Giá nhà")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x="sqft_living", y="price", hue="bedrooms", palette="cool", ax=ax)
    ax.set_title("Giá nhà theo diện tích sinh hoạt")
    st.pyplot(fig)

    # Biểu đồ thành phố
    if "city" in df.columns:
        st.markdown("### 🌆 Giá trung bình theo Thành phố")
        avg_price_by_city = df.groupby("city")["price"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        avg_price_by_city.plot(kind="barh", color="coral", ax=ax)
        ax.set_xlabel("Giá trung bình (USD)")
        ax.invert_yaxis()
        st.pyplot(fig)

    # Ma trận tương quan
    st.markdown("### 🧮 Ma trận tương quan giữa các đặc trưng")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[int, float]).corr(), annot=True, cmap="Spectral", fmt=".2f", ax=ax)
    st.pyplot(fig)

elif choice == "🤖 Dự đoán Giá Nhà":
    st.subheader("🤖 Nhập thông tin để dự đoán giá nhà")

    with st.form("form_dudoan"):
        col1, col2, col3 = st.columns(3)
        bedrooms = col1.slider("🛏️ Số phòng ngủ", 1, 10, 3)
        bathrooms = col2.slider("🛁 Số phòng tắm", 1, 8, 2)
        sqft_living = col3.number_input("📐 Diện tích sinh hoạt (sqft)", 300, 10000, 1500)
        floors = col1.selectbox("🏢 Số tầng", [1, 1.5, 2, 2.5, 3])
        condition = col2.slider("🏗️ Tình trạng nhà (1: tệ - 5: tốt)", 1, 5, 3)

        submit = st.form_submit_button("🚀 Dự đoán ngay")

        if submit:
            input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, floors, condition]],
                                      columns=["bedrooms", "bathrooms", "sqft_living", "floors", "condition"])
            prediction = model.predict(input_data)
            st.success(f"✅ Giá nhà dự đoán: ${prediction[0]:,.2f}")
            st.progress(min(int(prediction[0] / 1500000 * 100), 100))

st.caption("📌 Made with ❤️ by Khalam | Dữ liệu từ MongoDB | ML: RandomForest")
