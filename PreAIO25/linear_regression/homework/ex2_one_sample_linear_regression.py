import random


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
def implement_linear_regression(X_data , y_data , epoch_max = 50, lr = 1e-5):
    losses = []
    w1, w2, w3, b = initialize_params()
    N = len(y_data)

    for epoch in range(epoch_max):
        for i in range(N):
        # get a sample
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]
            y = y_data[i]
            # compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)
            # compute loss
            loss = compute_loss_mae(y, y_hat)
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

            losses.append(loss)
    return (w1 ,w2 ,w3 ,b, losses)

# y_hat = predict (x1 =1, x2 =1, x3 =1, w1 =0, w2 =0.5 , w3 =0, b =0.5)
# print(y_hat) # #Result Question2 A
# l = compute_loss_mse( y_hat =1, y =0.5)
# print (l) # #Result Question3 A
# g_wi = compute_gradient_wi(xi =1.0 , y=1.0 , y_hat =0.5)
# print(g_wi) # #Result Question4 A
# g_b = compute_gradient_b (y=2.0 , y_hat =0.5)
# print(g_b) # Result Question5 B
# after_wi = update_weight_wi (1.0 , -0.5 ,1e-5)
# print(after_wi) #Result Question6 A
# after_b = update_weight_b (b=0.5 , db = -1.0 , lr = 1e-5)
# print(after_b) #Result Question7 A

# X,y = prepare_data("advertising.csv")
# (w1 ,w2 ,w3 ,b, losses) = implement_linear_regression(X,y)
# plt.plot(losses[:100])
# plt.xlabel("#Iteration")
# plt.ylabel("Loss")
# plt.show()
#
# print(w1,w2,w3)
# ##Result Question 8 A
#
# #given new data
# tv = 19.2
# radio = 35.9
# newspaper = 51.3
# sales = predict(tv, radio, newspaper, w1, w2, w3, b)
# print(f"Predicted sales is {sales}")
#
# # #Result Question 9 B
# l = compute_loss_mae ( y_hat =1, y =0.5)
# print (l)
# # #Result Question 10 A