import torch
model = torch.load('../model.pt')
model.eval()
print("Num1:")
a=int(input())
print("Num2:")
b=int(input())
result=model(torch.FloatTensor([[a,b]]))
print(result)