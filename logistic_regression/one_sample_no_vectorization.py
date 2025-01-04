import numpy as np



def logistic_function(theta, x):
    z = np.dot(theta, x)
    y_hat = 1/ (1+ np.exp(-z)) #Sigmoid function
    return z, y_hat

def compute_loss(y, y_hat):
    loss = (-y * np.log(y_hat))-((1-y)*np.log(1-y_hat))
    return loss

def compute_derivative(x, y_hat, y):
    dL = x * (y_hat - y)
    return dL

def update_weight(theta, lr, dL):
    theta = theta - lr * dL
    return theta

data=np.genfromtxt("/Users/thachha/PycharmProjects/AIO24/logistic_regression/data.csv", delimiter=",",skip_header=1)
#print(data)
#print(type(data))

X = data[:, [0,1]] #array => 2 columns of feature
X = np.hstack((np.ones((X.shape[0], 1)), X)) # add bias column
Y = data[:, [2]] #label
#print(X)
no_sample = X.shape[0]
n_features = X.shape[1]
#print(Y)
print(f'Size of input: {X.shape}')
print(f'Number of sample: {no_sample}')
# print(f'Size of output: {Y.shape}')

theta = np.random.rand(n_features)
print(f'Initial weights:{theta}')
no_epoch = 10
lr = 0.001
losses=[]
for epoch in range(no_epoch):
    print(f"Epoch {epoch + 1}/{no_epoch}")
    for idx in range(no_sample):
        x = X[idx]
        #print(f"x: {x}")
        y = Y[idx]
        #print(f"y: {y}")
        z, y_hat = logistic_function(theta, x)
        #print(f"y_hat: {y_hat}")
        loss = compute_loss(y, y_hat)
        losses.append(loss)
        #print(f"loss: {loss}")
        dL = compute_derivative(x, y_hat, y)
        #print(f"dL: {dL}")
        theta = update_weight(theta, lr, dL)
    print(f"theta after {epoch}: {theta}")
    print("-" * 50)  # Separator for clarity between epochs
print(losses)

