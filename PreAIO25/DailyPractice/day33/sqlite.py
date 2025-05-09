import sqlite3

connection = sqlite3.connect('database.sqlite')
cursor = connection.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS CUSTOMERS (
        EMAIL TEXT PRIMARY KEY,
        NAME TEXT NOT NULL,
        PHONE TEXT NOT NULL
    );
""")

cursor.execute("""
    INSERT INTO CUSTOMERS (EMAIL, NAME, PHONE)
    VALUES
    ('nguyen@aivietnam.edu.vn', 'Nguyen', '123456789'),
    ('admin@aivietnam.edu.vn', 'Vinh', '1122334455');
""")

# Commit the changes and close the connection
connection.commit()

import pandas as pd
# Lấy tất cả data từ bảng CUSTOMER
data = pd.read_sql_query (" SELECT * FROM CUSTOMERS ", connection )
print( data )