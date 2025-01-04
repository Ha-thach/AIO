import numpy as np
import matplotlib.pyplot as plt
import random
from linear_regression.homework.ex1_advertising_regression import prepare_data

def predict(x1 , x2 , x3 , w1 , w2 , w3 , b):
    y_hat = w1 * x1 + w2 * x2 + w3 * x3 + b
    return y_hat

def compute_loss_mse(y_hat, y):
    loss = (y_hat - y) ** 2
    return loss

def compute_loss_mae(y_hat, y):
    loss= abs(y_hat - y)
    return loss
def compute_gradient_wi(xi, y, y_hat):
    dw = 2 * xi * (y_hat - y)
    return dw

def compute_gradient_b(y_hat, y):
    db = 2 * (y_hat - y)
    return db

def update_weight_wi(w, dw, lr):
    w = w - lr * dw
    return w

def update_weight_b(b, db, lr):
    b = b - lr * db
    return b

def initialize_params_should_use():
    w1 = random . gauss (mu =0.0 , sigma =0.01)
    w2 = random . gauss (mu =0.0 , sigma =0.01)
    w3 = random . gauss (mu =0.0 , sigma =0.01)
    b = 0
    return w1 , w2 , w3 , b
def initialize_params():
    w1 , w2 , w3 , b = (0.016992259082509283 , 0.0070783670518262355 , -0.002307860847821344 , 0)
    return w1 , w2 , w3 , b
def implement_linear_regression_nsamples(X_data , y_data , epoch_max = 50, lr = 1e-5):
    losses = []
    w1, w2, w3, b = initialize_params()
    N = len(y_data)

    for epoch in range(epoch_max):
        loss_total = 0.0
        dw1_total = 0.0
        dw2_total = 0.0
        dw3_total = 0.0
        db_total = 0.0
        for i in range(N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]
            y = y_data[i]
            # compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)
            # compute loss
            loss = compute_loss_mse(y, y_hat)
            loss_total += loss
            # compute gradient w1 , w2 , w3 , b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)
            # update parameters
            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)
            # accumulate gradient w1 , w2 , w3 , b
            dw1_total +=dl_dw1
            dw2_total += dl_dw2
            dw3_total += dl_dw3
            db_total += dl_db

        dw1_total = dw1_total/N
        dw2_total = dw2_total/N
        dw3_total = dw3_total/N
        db_total = db_total/N
        loss_total=loss_total/N
        #logging
        losses.append(loss_total)
    return (w1 ,w2 ,w3 ,b, losses)

X,y=prepare_data("advertising.csv")
(w1 ,w2 ,w3 ,b, losses ) = implement_linear_regression_nsamples (X, y,
epoch_max =1000 ,lr =1e-5)
print( losses )
plt.plot( losses )
plt.xlabel("# epoch ")
plt.ylabel("MSE Loss ")
plt.show()


