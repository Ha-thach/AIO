import numpy as np

#EX1
# data = [3 , 7 , 8 , 5 , 12 , 14 , 21 , 13 , 18]
#
# data.sort()
# print(data)
#
# # Calculate Q1 and Q3
# Q1 = np.percentile(data, 25)
# Q3 = np.percentile(data, 75)
#
# # Calculate IQR
# IQR = Q3 - Q1
#
# print(f"Q1: {Q1}")
# print(f"Q3: {Q3}")
# print(f"IQR: {IQR}")

# #EX2
# data = [3, 5, 7, 8, 12, 13, 14, 18, 21, 100]
#
# # Tính Q1, Q3
# Q1 = np.percentile(data, 25)  # Q1 - Phân vị thứ 25%
# Q3 = np.percentile(data, 75)  # Q3 - Phân vị thứ 75%
#
# # Tính IQR
# IQR = Q3 - Q1
#
# # Xác định ngưỡng dưới và trên
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# # Tìm các giá trị ngoại lai
# outliers = [x for x in data if x < lower_bound or x > upper_bound]
#
# # In kết quả
# print(f"Q1: {Q1}")
# print(f"Q3: {Q3}")
# print(f"IQR: {IQR}")
# print(f"Ngưỡng dưới: {lower_bound}")
# print(f"Ngưỡng trên: {upper_bound}")
# print(f"Các giá trị ngoại lai: {outliers}")

#EX3
import pandas as pd
import numpy as np

# Tạo DataFrame
df = pd.DataFrame({
    'score': [55, 61, 70, 65, 68, 90, 91, 94, 300, 58]
})

# Tính Q1, Q3
Q1 = np.percentile(df['score'], 25)  # Phân vị 25%
Q3 = np.percentile(df['score'], 75)  # Phân vị 75%

# Tính IQR
IQR = Q3 - Q1

# Xác định ngưỡng dưới và trên
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Xác định các giá trị ngoại lai
outliers = df[(df['score'] < lower_bound) | (df['score'] > upper_bound)]
print("Các giá trị ngoại lai:")
print(outliers)

# Tạo DataFrame mới không chứa outlier
df_clean = df[(df['score'] >= lower_bound) & (df['score'] <= upper_bound)]

# In DataFrame sau khi loại bỏ outliers
print("\nDataFrame sau khi loại bỏ outlier:")
print(df_clean)
