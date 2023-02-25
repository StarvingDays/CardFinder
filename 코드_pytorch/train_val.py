import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def train(epoch, model, device, 
        data_loader, optimizer, 
        dataset_size, best_mse, model_name):

    model.train()
    kLogInterval = 10
    mse = 0.0
    count = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = Variable(data)
        target = Variable(target)   
        data, target = data.to(device), target.to(device)
        

        optimizer.zero_grad()
        output = model.forward(data)
        loss = F.cross_entropy(output, target)
        
        mse += loss.item()

        loss.backward()
        optimizer.step()

        if(batch_idx % kLogInterval == 0):
            
            print(f'Train Epoch : {epoch} [{batch_idx * data.size(0)}/{dataset_size}] Loss: {loss.item()}')
            count += 1


    mse /= float(count)
    print(" mean squared error: %f\n", mse)

    if(mse < best_mse):
        separator = './'
        print('save!')
        

        traced = torch.jit.trace(model, data_loader.dataset[0][0].unsqueeze(0))
        traced.save(separator + model_name + '_' +str(mse)+'.pt')

        best_mse = mse

        
    

def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data)
            target = Variable(target)  
            data, target = data.to(device), target.to(device)
            output = model.forward(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            print("target", target)
            print("pred", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("correct", correct)
            break

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy