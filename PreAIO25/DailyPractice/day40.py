# import matplotlib.pyplot as plt
#
# # Dữ liệu
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 25, 30, 40]
#
# plt.plot(x, y, marker='o', linestyle='-', color='b', label="Dữ liệu")
#
# plt.xlabel("Trục X")
# plt.ylabel("Trục Y")
# plt.title("Biểu đồ đường với Matplotlib")
#
# # Hiển thị chú thích
# plt.legend()
#
# # Hiển thị biểu đồ
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Dữ liệu giả lập
# data = [10, 20, 25, 30, 40, 35, 50, 60, 55]
#
# # Vẽ biểu đồ phân bố với Seaborn
# sns.histplot(data, kde=True, color='blue')
#
# # Thêm tiêu đề
# plt.title("Biểu đồ phân bố với Seaborn")
#
# # Hiển thị biểu đồ
# plt.show()
import pandas as pd

# Đọc file CSV
df = pd.read_csv("/Users/thachha/Downloads/menu.csv")

print(df.head())       # Hiển thị 5 dòng đầu tiên
print(df.tail())       # Hiển thị 5 dòng cuối cùng
print(df.info())       # Thông tin chung về dataframe
print(df.describe())   # Thống kê mô tả dữ liệu
