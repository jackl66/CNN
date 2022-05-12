import argparse as ap
from efficientnet_ball import *


 
parser = ap.ArgumentParser()
parser.add_argument('epoch', help="epoch", choices={'200', '400', '600', '1000'})

parser.add_argument('lr', help="learning rate", choices={'0.000025', '0.0001', '0.0002', '0.0004','0.001'})
parser.add_argument('cuda', help="use which gpu", choices={'0', '1'})

parser.add_argument('batch', help="batch size", choices={'16', '32', '64', '128'})
parser.add_argument('model', help='raw or refine', choices={'0', '1','2','3'})

args = parser.parse_args()

batch_size = int(args.batch)
network_type = int(args.model)
lr = float(args.lr)
device = args.cuda
epoch = int(args.epoch)
patience = 25

cuda_idx = 'cuda:' + device
device = T.device(cuda_idx if T.cuda.is_available() else 'cpu')
print("Train on {}".format(device))

# build the desired model
Net_builder = EfficientNet_Model(model_idx=network_type)

# load images into pytorch loader
trainset = datasets.ImageFolder(Net_builder.train_path, transform=Net_builder.train_transform)
print("Trainset Size:  {}".format(len(trainset)))

validateset = datasets.ImageFolder(Net_builder.validate_path, transform=Net_builder.validate_transform)
print("validateset Size:  {}".format(len(validateset)))

testset = datasets.ImageFolder(Net_builder.test_path, transform=Net_builder.validate_transform)
print("testset Size:  {}".format(len(testset)))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
print("No. of batches in trainloader:{}".format(len(trainloader)))
print("No. of Total examples:{}".format(len(trainloader.dataset)))

validationloader = DataLoader(validateset, batch_size=batch_size, shuffle=False, num_workers=2)
print("No. of batches in validationloader:{}".format(len(validationloader)))
print("No. of Total examples:{}".format(len(validationloader.dataset)))

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
print("No. of batches in testloader:{}".format(len(testloader)))
print("No. of Total examples:{}".format(len(testloader.dataset)))

# plot some example
# class_name = np.arange(6, dtype=int)

# img, label = trainset[20]
# print(trainset.class_to_idx)
# show_image(img, class_name[label])

# img, label = trainset[26]
# print(trainset.class_to_idx)
# show_image(img, class_name[label])


model = Net_builder.create_new_top_layer()
criterion = nn.CrossEntropyLoss()
optimizer = T.optim.Adam(model.parameters(), lr=lr)
# model.load_state_dict(T.load('./models/tf_efficientnet_b4/1648950309', map_location="cuda:1"),strict=False)

model = model.to(device)

# logging info
training_accuracy = []
training_loss = []
validation_accuracy = []
validation_loss = []
early_stopping = []
best = 0
print("model created & starting training ...\n\n")

for i in range(epoch):
    correct_preds = 0.0
    total_train = 0
    train_loss = 0.0
    # step loop for training, training by batch
    model.train()
    for step, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        state_loss = criterion(output, labels)
      
        optimizer.zero_grad()
        state_loss.backward()
        optimizer.step()
        train_loss += state_loss.item()
        _,pred_idx = T.max(output,dim=1)
        correct_preds += T.sum(pred_idx == labels).item()
        total_train+=labels.size(0)
    epoch_accuracy = round(correct_preds / total_train, 4)
    epoch_loss = round(train_loss / total_train, 4)
    training_accuracy.append(epoch_accuracy)
    training_loss.append(epoch_loss)
    print('epoch {}: accuracy: {}, loss: {}\n'.format(i, epoch_accuracy, epoch_loss), end="")

    # run on validation set
    model.eval()
    val_loss = 0.0
    total = 0
    val_correct_preds = 0

    # validation
    with T.no_grad():
        for step, (images, labels) in enumerate(validationloader):
            # move all data loaded from dataloader to gpu
            images = images.to(device)
            labels = labels.to(device)

            # feed-forward data in the model
            output = model(images)

            # compute losses
            state_loss = criterion(output, labels)
            # aggregate loss for logging
            val_loss += state_loss.item()

            # accuracy computation
            _,pred_idx = T.max(output,dim=1)
            val_correct_preds += T.sum(pred_idx == labels).item()
            total +=labels.size(0)
    val_accuracy = round(val_correct_preds / total, 4)
    val_loss = round(val_loss / total, 4)

    print('\nvalidation {}: accuracy: {}, loss: {}\n\n'.format(i, val_accuracy, val_loss), end="")

    # save best model
    validation_accuracy.append(val_accuracy)
    validation_loss.append(val_loss)
    is_best = val_accuracy > best
    if is_best:
        best = val_accuracy
        Net_builder.save_checkpoint(model.state_dict())

    # early stopping check
    early_stopping.append(val_accuracy)
    ind = early_stopping.index(best)
    if len(early_stopping) - ind > patience:
        print('patience has ran out,  best val accuracy {}'.format(best))
        break

# log training processes
dir_path = './npy/' + Net_builder.token
os.mkdir(dir_path)
path1 = os.path.join(dir_path, 'train_acc.npy')
path2 = os.path.join(dir_path, 'val_acc.npy')
path3 = os.path.join(dir_path, 'train_loss.npy')
path4 = os.path.join(dir_path, 'val_lose.npy')
path5 = os.path.join(dir_path, 'predicts.npy')
path6 = os.path.join(dir_path, 'gt.npy')


with open(path1, 'wb') as f:
    np.save(f, np.array(training_accuracy))
with open(path2, 'wb') as f:
    np.save(f, np.array(validation_accuracy)) 
with open(path3, 'wb') as f:
    np.save(f, np.array(training_loss))
with open(path4, 'wb') as f:
    np.save(f, np.array(validation_loss))

# testing  
model.load_state_dict(T.load(Net_builder.file_path, map_location="cuda:1"),strict=False)
model=model.to(device)
model.eval()

correct = 0
total = 0
predictions =[]
gt= []
# since we're not training, we don't need to calculate the gradients for our outputs
with T.no_grad():
    for images,labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
         
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = T.max(outputs.data, 1)
        predictions.append(predicted.cpu().detach().numpy().tolist())
        gt.append(labels.cpu().detach().numpy().tolist())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print(Net_builder.token,Net_builder.model_name,args.model,args.batch,args.lr,best)

with open(path5, 'wb') as f:
    np.save(f, np.array(predictions,dtype=object))
with open(path6, 'wb') as f:
    np.save(f, np.array(gt,dtype=object))