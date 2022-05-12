import argparse as ap
from efficientnet_ball import *

# this file will only load the model and run testing
 
parser = ap.ArgumentParser()
# parser.add_argument('epoch', help="epoch", choices={'200', '400', '600', '1000'})

parser.add_argument('lr', help="learning rate", choices={'0.000025', '0.0001', '0.0002', '0.0004','0.001'})
parser.add_argument('cuda', help="use which gpu", choices={'0', '1'})

parser.add_argument('batch', help="batch size", choices={'16', '32', '64', '128'})
parser.add_argument('model', help='raw or refine', choices={'0', '1','2','3'})

args = parser.parse_args()

batch_size = int(args.batch)
network_type = int(args.model)
lr = float(args.lr)
device = args.cuda
# patience = 25

cuda_idx = 'cuda:' + device
device = T.device(cuda_idx if T.cuda.is_available() else 'cpu')
print("Testing on {}".format(device))

# build the desired model
Net_builder = EfficientNet_Model(model_idx=network_type)

# load images into pytorch loader
testset = datasets.ImageFolder(Net_builder.test_path, transform=Net_builder.validate_transform)
print("testset Size:  {}".format(len(testset)))

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
print("No. of batches in testloader:{}".format(len(testloader)))
print("No. of Total examples:{}".format(len(testloader.dataset)))


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
print("model created & starting testing ...\n")

# now it will load b0 and test on refine dataset
model_path = './models/tf_efficientnet_b0/1651339309'

# testing  
model.load_state_dict(T.load(model_path, map_location="cuda:1"),strict=False)
model=model.to(device)
model.eval()

correct = 0
total = 0
predictions =[]
gt= []
# log testing processes
dir_path = './npy/' + Net_builder.token
os.mkdir(dir_path)
path5 = os.path.join(dir_path, 'predicts.npy')
path6 = os.path.join(dir_path, 'gt.npy')
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