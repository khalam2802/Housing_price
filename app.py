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
model = joblib.load("house_price_rf_model.pkl")

# Tabs giao diện
menu = ["📊 Thống kê & Biểu đồ", "📂 Dữ liệu", "🤖 Dự đoán Giá Nhà"]
choice = st.sidebar.radio("Chọn chức năng:", menu)

if choice == "📂 Dữ liệu":
    st.subheader("📂 Dữ liệu từ MongoDB")
    st.dataframe(df.head(200))

elif choice == "📊 Thống kê & Biểu đồ":
    st.subheader("📊 Thống kê mô tả dữ liệu")

    col1, col2, col3 = st.columns(3)
    col1.metric("🏘️ Số lượng nhà", df.shape[0])
    col2.metric("💲 Giá trung bình", f"{df['price'].mean():,.0f} USD")
    col3.metric("📐 Diện tích TB", f"{df['sqft_living'].mean():,.0f} sqft")

    st.markdown("### 🔹 Biểu đồ phân phối giá nhà")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], kde=True, color="skyblue", ax=ax1)
    ax1.set_xlabel("Giá nhà (USD)")
    st.pyplot(fig1)

    st.markdown("### 🔹 Biểu đồ hộp giá theo số phòng ngủ")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="bedrooms", y="price", data=df, palette="Set2", ax=ax2)
    ax2.set_xlabel("Số phòng ngủ")
    ax2.set_ylabel("Giá nhà (USD)")
    st.pyplot(fig2)

    st.markdown("### 🔹 Giá trung bình theo thành phố")
    if "city" in df.columns:
        avg_price_by_city = df.groupby("city")["price"].mean().sort_values(ascending=False).head(10)
        fig3, ax3 = plt.subplots()
        avg_price_by_city.plot(kind="bar", color="orange", ax=ax3)
        ax3.set_ylabel("Giá trung bình (USD)")
        ax3.set_xlabel("Thành phố")
        st.pyplot(fig3)

    st.markdown("### 🔹 Ma trận tương quan các đặc trưng")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[int, float]).corr(), annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

elif choice == "🤖 Dự đoán Giá Nhà":
    st.subheader("🤖 Nhập thông tin để dự đoán giá nhà")

    bedrooms = st.slider("Số phòng ngủ", 1, 10, 3)
    bathrooms = st.slider("Số phòng tắm", 1, 8, 2)
    sqft_living = st.number_input("Diện tích sinh hoạt (sqft)", 300, 10000, 1500)
    floors = st.selectbox("Số tầng", [1, 1.5, 2, 2.5, 3])
    condition = st.slider("Tình trạng nhà (1: tệ - 5: tốt)", 1, 5, 3)

    if st.button("Dự đoán ngay"):
        input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, floors, condition]],
                                  columns=["bedrooms", "bathrooms", "sqft_living", "floors", "condition"])
        prediction = model.predict(input_data)
        st.success(f"✅ Giá nhà dự đoán: ${prediction[0]:,.2f}")
        st.progress(min(int(prediction[0] / 1500000 * 100), 100))

st.caption("Made with ❤️ by Khalam | Dữ liệu từ MongoDB | Dự đoán với RandomForest")
