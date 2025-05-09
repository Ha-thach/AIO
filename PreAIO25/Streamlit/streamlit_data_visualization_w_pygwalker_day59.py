import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import pygwalker as pyg

st.title ("Phân tích dữ liệu từ file CSV với PyGWalker ")
uploaded_file = st.file_uploader("Chọn file CSV", type =["csv"])
if uploaded_file is not None :
    df=pd.read_csv(uploaded_file)
    st.subheader("Dữ liệu ban đầu")
    st.dataframe(df.head())
    st.write("Số dòng:", df.shape[0])
    st.write("Số cột:", df.shape[1])
st.divider()
if uploaded_file is not None :
    st.subheader ("Thông tin mô tả dữ liệu")
    st.write("Các thống kê cơ bản:")
    st.dataframe(df.describe())
    st.write("Kiểu dữ liệu:",df. dtypes)
    st.write("Khảo sát giá trị NULL:")
    st.write(df.isnull().sum())
st.divider()

if uploaded_file is not None:
    st.subheader("📊 Phân tích dữ liệu tương tác với PygWalker")

    pyg_html = pyg.walk(df, return_html=True)
    # Hiển thị trong ứng dụng Streamlit
    components.html(pyg_html, height=1000, scrolling=True)