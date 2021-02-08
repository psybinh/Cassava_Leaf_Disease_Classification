import torch
import tqdm
import time

def train_one_epoch(epoch,
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    lr_scheduler,
                    gradient_to_accumulation=1):

    scaler = torch.cuda.amp.GradScaler()

    tbar = tqdm.tqdm(train_loader)
    total = 0
    correct = 0
    running_loss = 0.0
    for i, data in enumerate(tbar):
        optimizer.zero_grad()
        inputs, labels = data
        with torch.cuda.amp.autocast():
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss /= gradient_to_accumulation
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scaler.scale(loss).backward()

        if (i + 1) % gradient_to_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            tbar.set_description(
                'Epoch: {}, train loss: {:.4f}, train acc: {:.4f}'.format(epoch,
                                                                          running_loss / (i + 1),
                                                                          correct / total))

    lr_scheduler.step()

    return running_loss / len(train_loader) * gradient_to_accumulation, correct / total

def validate(epoch, model, val_loader,  criterion, exp_name=None, current_best_acc=1):
    tbar = tqdm.tqdm(val_loader)
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(tbar):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
        
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()

            tbar.set_description(
                'Epoch: {}, val loss: {:.4f}, val acc: {:.4f}'.format(epoch,
                                                                      running_loss / (i + 1),
                                                                      correct / total))

    if correct / total > current_best_acc and exp_name != None:
        torch.save(model.state_dict(),
                   'saved_models/exp_{}_time_{}_epoch_{}_acc_{}.pth'.format(exp_name,
                                                                      int(time.time()),
                                                                      epoch,
                                                                      correct / total))
        current_best_acc = correct / total

    return running_loss / len(val_loader), correct / total, current_best_acc