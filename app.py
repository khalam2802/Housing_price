import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pymongo import MongoClient

st.set_page_config(layout="wide")
st.title("🏠 Phân tích & Dự đoán Giá Nhà từ MongoDB")

# Kết nối MongoDB
client = MongoClient("mongodb+srv://khalam28:khalam28@cluster0.je2xxuc.mongodb.net")
db = client["housing_price"]
collection = db["data"]

# Load dữ liệu từ MongoDB
data = list(collection.find({}, {"_id": 0}))  # bỏ _id cho dễ xử lý
df = pd.DataFrame(data)

# Load model
model = joblib.load("house_price_rf_model.pkl")

tab1, tab2, tab3 = st.tabs(["📂 Dữ liệu", "📊 Thống kê", "🤖 Dự đoán"])

with tab1:
    st.subheader("📂 Dữ liệu từ MongoDB")
    st.dataframe(df.head(100))

with tab2:
    st.subheader("📊 Thống kê mô tả")
    st.write(df.describe())

    st.write("Biểu đồ phân phối giá nhà")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.write("Biểu đồ giá nhà theo số phòng ngủ")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="bedrooms", y="price", data=df, ax=ax2)
    st.pyplot(fig2)

with tab3:
    st.subheader("🤖 Dự đoán giá nhà")

    bedrooms = st.slider("Số phòng ngủ", 1, 10, 3)
    bathrooms = st.slider("Số phòng tắm", 1, 8, 2)
    sqft_living = st.number_input("Diện tích sinh hoạt (sqft)", 300, 10000, 1500)
    floors = st.selectbox("Số tầng", [1, 1.5, 2, 2.5, 3])
    condition = st.slider("Tình trạng nhà (1-5)", 1, 5, 3)

    if st.button("Dự đoán"):
        input_data = [[bedrooms, bathrooms, sqft_living, floors, condition]]
        prediction = model.predict(input_data)
        st.success(f"✅ Giá nhà dự đoán: ${prediction[0]:,.2f}")
