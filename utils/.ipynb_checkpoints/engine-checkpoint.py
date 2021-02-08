
def train_one_epoch(epoch, model, train_loader, criterion, optimizer, finetune_on_bn=True, logs_file=None):
    if finetune_on_bn == False:
        freeze_bn(model)
    else:
        model.train()
        
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()
    for i, data in enumerate(tqdm.tqdm(train_loader), 0):
        inputs, labels = data
        with torch.cuda.amp.autocast():
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            assert outputs.dtype is torch.float16
            loss = criterion(outputs, labels)
            loss = loss / GRADIENT_COEFF
            assert loss.dtype is torch.float32
        
        scaler.scale(loss).backward()
        
        if (i + 1) % GRADIENT_COEFF == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running_loss += loss.item()
            
            if logs_file != None:
                logs_file.write('Training loss at epoch {}, iteration {}: {}\n'.format(epoch, i, running_loss / i * GRADIENT_COEFF * BATCH_SIZE))
    if logs_file != None:
        logs_file.write('------------------------------------------------\n')
           
    return running_loss / len(train_loader)

def validate(epoch, model, val_loader, logs_file=None):
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in tqdm.tqdm(val_loader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
        
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
    
    if logs_file != None:
        logs_file.write('Validating loss at epoch {}: {}\n'.format(epoch, running_loss / len(val_loader)))
        logs_file.write('Validating accuracy at epoch {}: {}\n'.format(epoch, correct / total))
        logs_file.write('**************************************************\n')
        
    torch.save(model.state_dict(), 'models/exp_{}_time_{}_epoch_{}_acc_{}.pth'.format(EXP_NAME, int(time.time()), epoch, correct / total))
    return running_loss / len(val_loader), correct / total