import numpy as np
import matplotlib.pyplot as plt
import random

def get_column(data, index):
    return [row[index] for row in data]

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',',
                         skip_header=1).tolist()
    N=len(data)

    # get tv (index=0)
    tv_data = get_column(data, 0)
    #print(f'TV: {tv_data[:5]}')
    # get radio (index=1)
    radio_data = get_column(data, 1)
    #print(f'Radio: {radio_data[:5]}')

    # get newspaper (index=2)
    newspaper_data = get_column(data, 2)
    #print(f'Newspaper: {newspaper_data[:5]}')

    # get sales (index=0)
    sales_data = get_column(data, 3)
    #print(f'Sales: {sales_data[:5]}')

    # building X input and y output for training
    X = [tv_data, radio_data, newspaper_data]
    y = sales_data
    return X, y

X,y = prepare_data ("advertising.csv")
#list = [sum(X[0][:5]) , sum(X[1][:5]) , sum(X[2][:5]) , sum(y[:5])]
#print(list)

#Result A