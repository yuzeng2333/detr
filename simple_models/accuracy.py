import torch

def count_accuracy(output, targets):
    """output is of shape (batch_size, n_classes)"""        
    output = output.view(-1, output.shape[-1])
    pred = output.argmax(dim=1, keepdim=True)
    # flatten pred
    pred = pred.view(-1)
    degrees = []
    for target in targets:
        degrees.append(target['max_degree'])
    # flatten the first dimension of degrees
    degrees = torch.tensor(degrees).view(-1)
    correct = pred.eq(degrees.view_as(pred)).sum().item()
    return correct