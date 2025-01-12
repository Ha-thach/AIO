#import library
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""Data loading, train_slit test and preprocessing using StandardScalar, convert from numpy to tensor """
data =load_iris()
#print(data)
print(data.keys()) #dict_keys(['data', 'target', 'frame', '...
print(type(data)) #<class 'sklearn.utils._bunch.Bunch'>
print(data.data.shape) #(150, 4)
print(data.target.shape) #(150,)

X_train, X_test, Y_train, Y_test = train_test_split(
    data.data,
    data.target,
    test_size= 0.4,
    random_state = 7
)

X_valid, X_test, Y_valid, Y_test = train_test_split(
    X_test,
    Y_test,
    test_size= 0.5,
    random_state = 7
)

print(X_train.shape, X_valid.shape, X_test.shape) #(90, 4) (30, 4) (30, 4)
print(Y_train.shape, Y_valid.shape, Y_test.shape) #(90,) (30,) (60,)
print(X_train)
#data_processing
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_valid = scalar.transform(X_valid)
X_test = scalar.transform(X_test)
# print("-------------")
# print(X_train)
# print(type(X_train)) #<class 'numpy.ndarray'>

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
Y_valid = torch.tensor(Y_valid)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test)

#design MLP model
model = nn.Sequential(
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4,3),
)
print(model)
#summary(model, (1000,4))

for p in model.parameters():
    nn.init.constant_(p, 0.1)

for layer in model.children():
    print(layer.state_dict())
#print(x)
#print(y)
print(X_train[0])
print(Y_train[0])
loss_fn = nn.CrossEntropyLoss() #please notice target tensor type in lossfunction to define correctly input
y_pred= model(X_train[0])
print(y_pred)
loss = loss_fn(y_pred, Y_train[0]) #requires long scalar here!

#init parameter
lr=0.1
optimizer = optim.SGD(model.parameters(), lr)

def evaluate(model, X_valid, Y_valid):
    with torch.no_grad():
        Y_pred = model(X_valid)
    Y_pred = torch.argmax(Y_pred, dim=1)
    return sum(Y_pred == Y_valid)/len(Y_valid)
evaluate(model, X_valid, Y_valid)
print(evaluate)

num_epochs = 20
losses = []
best_acc = 0
accs =[]
for epoch in range(num_epochs):
    epoch_loss = []
    for x_train, y_train in zip(X_train, Y_train):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        epoch_loss.append(loss.item())
        optimizer.zero_grad() #resets the gradients of all model parameters to zero to avoid gradient accumulation. gradient khac parameter!!!!
        loss.backward()
        optimizer.step()
    avg_loss = sum(epoch_loss)/len(epoch_loss)
    losses.append(avg_loss)
    acc_valid = evaluate(model, X_valid, Y_valid)
    accs.append(acc_valid)
    if  len(accs)>=4:
        if (accs[-1] <= accs[-3] and len(accs) >=4):
            break
    print(f"Epoch {epoch}: Average_loss: {avg_loss} -- Accuracy:{acc_valid}")

#test
with torch.no_grad():
    Y_pred = model(X_test)
    Y_pred = torch.argmax(Y_pred, dim=1)
    print(Y_pred == Y_test)
    print(sum(Y_pred==Y_test)/len(X_test))

#Depends on the loss and acc in valid and test to modify model which can learn more features.