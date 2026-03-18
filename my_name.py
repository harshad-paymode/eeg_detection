import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Hi My name is Harshad and I am running Kaggle from local enviornment")