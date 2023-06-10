def count_accuracy(output, target):
    """output is of shape (batch_size, n_classes)"""
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct