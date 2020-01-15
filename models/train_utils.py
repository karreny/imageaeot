import torch
import torch.optim as optim

def train_model(trainloader, model, optimizer, target_kw='label', single_batch=False):
    '''Method for training model (updating model params) based on given criterion'''
    
    model.train()

    total_loss = 0
    total_samples = 0

    for sample in trainloader:
        model.zero_grad()
        output = model(sample)

        target = sample[target_kw]
        batch_size = len(target)

        loss = model.compute_loss(output)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()*batch_size
        total_samples += batch_size

        if single_batch:
            break

    return {'train_loss': total_loss / total_samples}

def evaluate_model(testloader, model, target_kw='label', single_batch=False):
    '''Method for evaluating model based on given criterion'''
    
    model.eval()

    total_loss = 0
    total_samples = 0

    for sample in testloader:
        with torch.no_grad():
            output = model(sample)
            target = sample[target_kw]
            batch_size = len(target)

            loss = model.compute_loss(output)

        total_loss += loss.item()*batch_size
        total_samples += batch_size

        if single_batch:
            break

    return {'test_loss': total_loss / total_samples}

def save_checkpoint(current_state, filename):
    torch.save(current_state, filename)

def setup_optimizer(name, param_list):
    if name == 'sgd':
        return optim.SGD(param_list, momentum=0.9)
    elif name == 'adam':
        return optim.Adam(param_list)
    else:
        raise KeyError("%s is not a valid optimizer (must be one of ['sgd', adam']" % name)
