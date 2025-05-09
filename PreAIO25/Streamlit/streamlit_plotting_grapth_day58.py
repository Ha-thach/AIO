"""
Exercise 1: Plotting Basic Functions
Requirement: Select one of the functions sin, cos, exp, or log, and plot it over the interval [−10, 10].

Exercise 2: Comparing Two Functions on the Same Chart
Requirement: Choose any two functions from sin, cos, exp, or log, and plot them together on the same chart.

Exercise 3: Plotting a Quadratic Function
Requirement: Enter the coefficients a, b, and c for the equation y = ax² + bx + c, and plot the corresponding graph.

Exercise 4: Interactive Plotting with Sliders
Requirement: Use st.slider to adjust the values of a, b, and c, and update the graph in real time.

Exercise 5: Plotting a Heatmap for the Function z = x² + y²
Requirement: Use sns.heatmap to draw the heatmap for the function z = x² + y².
"""
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def set_funct(funct, x):
    if funct == "sin":
        y = np.sin(x)
    elif funct == "cos":
        y = np.cos(x)
    elif funct == "exp":
        y = np.exp(x)
    elif funct == "log":
        y = np.log(x)
    return y

def quadratic_funct(a, b, c, x):
    return (a * x * x + b * x + c)

st.title("Bài tập Khảo sát biểu đồ - Ngày 58")
st.divider()

bt=st.sidebar.selectbox("Chọn bài tập:",
                        ["Bài 1: Vẽ đồ thị hàm số cơ bản",
                         "Bài 2: So sánh 2 hàm số trên cùng một biểu đồ",
                         "Bài 3: Vẽ đồ thị hàm bậc 2",
                         "Bài 4: Tương tác với Slider để khảo sát đồ thị",
                         "Bài 5: Vẽ Heatmap cho hàm z = x^2 + y^2"])
if bt == "Bài 1: Vẽ đồ thị hàm số cơ bản":
    st.subheader("Bài 1: Vẽ đồ thị hàm số cơ bản")
    st.text("Yêu cầu: Chọn 1 trong các hàm số sin, cos, exp, log và vẽ biểu đồ trên đoạn [−10, 10]:")
    funct=st.selectbox("Chọn hàm số",
                        ["sin",
                         "cos",
                         "exp",
                         "log",])

    st.write(f"Đây là biểu đồ của đồ thị hàm số {funct}(x) ")

    x = np.arange(-10, 10, 0.1)
    if funct == "log":
        x = np.arange(0.1, 10, 0.1)
    y = set_funct(funct,x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.grid()
    ax.set_title(f"Đồ thị hàm số f(x)={funct}(x)")
    ax.set_xlabel("x")
    ax.set_ylabel(f"{funct}(x)")
    st.pyplot(fig)
elif bt == "Bài 2: So sánh 2 hàm số trên cùng một biểu đồ":
    st.subheader("Bài 2: So sánh 2 hàm số trên cùng một biểu đồ")
    funct = st.write("Yêu cầu: Chọn hai hàm số bất kỳ trong số: sin, cos, exp, log, và vẽ chúng trên cùng một biểu đồ.")
    functions = ["sin", "cos", "exp", "log"]
    y1 = st.selectbox("Hàm số thứ nhất:", functions)
    y2_options = [f for f in functions if f != y1]
    y2 = st.selectbox("Hàm số thứ hai:", y2_options)

    x = np.arange(-10, 10, 0.1)
    st.write(f"Đây là biểu đồ của hàm số {y1}(x) và {y2}(x) ")
    if y1 == "log" or y2 == "log":
        x = np.arange(0.1, 10, 0.1)
    y_1 = set_funct(y1, x)
    y_2 = set_funct(y2, x)

    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(x, y_1, label=f'f(x)={y1}(x)', color='blue')
    ax.plot(x, y_2, label=f'g(x)={y2}(x)', color='orange')

    ax.set_title(f'So sánh {y1}(x) (xanh) và {y2}(x) (cam)')
    ax.set_xlabel('x')
    ax.set_ylabel('g(x), f(x)')
    ax.legend(loc='upper left')
    st.pyplot(fig)

elif bt == "Bài 3: Vẽ đồ thị hàm bậc 2":
    st.subheader("Bài 3: Vẽ đồ thị hàm bậc 2")
    st.write("Yêu cầu: Nhập hệ số a, b, c cho phương trình y = ax^2 + bx + c và vẽ đồ thị tương ứng.")
    a = st.number_input("Nhập hệ số a:")
    b = st.number_input("Nhập hệ số b:")
    c = st.number_input("Nhập hệ số c:")

    x = np.arange(-10, 10, 0.1)
    y = quadratic_funct(a, b, c, x)
    st.write("a=", a)
    st.write("b=", b)
    st.write("c=", c)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='Đồ thị hàm số bậc hai y=ax^2 + bx + c')
    ax.grid()
    ax.set_title(f"Đồ thị hàm số bậc hai y={a}x^2 + {b}x + {c}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    st.pyplot(fig)
elif bt == "Bài 4: Tương tác với Slider để khảo sát đồ thị":
    st.subheader("Bài 4: Tương tác với Slider để khảo sát đồ thị")
    st.write("Dùng slider để điều chỉnh giá trị của a, b, c và cập nhật đồ thị theo thời gian thực.")

    a = st.slider("Nhập hệ số a", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    b = st.slider("Nhập hệ số b:",min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    c = st.slider("Nhập hệ số c:", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)

    x = np.arange(-10, 10, 0.1)
    y = quadratic_funct(a, b, c, x)
    st.write("a=", a)
    st.write("b=", b)
    st.write("c=", c)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='Đồ thị hàm số bậc hai y=ax^2 + bx + c')
    ax.grid()
    ax.set_title(f"Đồ thị hàm số bậc hai y={a}x^2 + {b}x + {c}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Tính và hiển thị đỉnh/đáy của parabol
    if a != 0:
        x_cuctri = -b / (2 * a)
        y_cuctri = a * x_cuctri ** 2 + b * x_cuctri + c
        ax.plot(x_cuctri, y_cuctri, 'ro')
        st.write(f"Điểm cực trị: ({x_cuctri:.2f}, {y_cuctri:.2f})")

        # Tính điểm cắt trục x
        delta = b ** 2 - 4 * a * c
        if delta > 0:
            x1 = (-b + np.sqrt(delta)) / (2 * a)
            x2 = (-b - np.sqrt(delta)) / (2 * a)
            st.write(f"Điểm cắt trục x: x₁ = {x1:.2f} và x₂ = {x2:.2f}")
        elif delta == 0:
            x0 = -b / (2 * a)
            st.write(f"Tiếp xúc trục x tại x = {x0:.2f}")
        else:
            st.write("Không có điểm cắt trục x")


    st.pyplot(fig)
elif bt == "Bài 5: Vẽ Heatmap cho hàm z = x^2 + y^2":
    st.subheader("Bài 5: Vẽ Heatmap cho hàm z = x^2 + y^2")
    st.write("Yêu cầu: Dùng sns.heatmap để vẽ biểu đồ nhiệt của hàm z = x^2 + y^2")

    x = np.arange(-10, 10, 0.1)
    y = np.arange(-10, 10, 0.1)

    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(Z)

    ax.set_title("Biểu đồ nhiệt của hàm z = x² + y²")
    st.pyplot(fig)
