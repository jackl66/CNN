import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import seaborn as sn
import pandas as pd

path1='./npy/1651340164/train_acc.npy'
data1=np.load(path1)

path2='./npy/1651340164/val_acc.npy'
data2=np.load(path2)
 
path3='./npy/1651340164/train_loss.npy'
data3=np.load(path3)   
 
path4='./npy/1651340164/val_lose.npy'
data4=np.load(path4)


fig = plt.figure()
fig.set_size_inches(10.5, 6.5)

plt.subplot(1,2, 1)
plt.plot(data1, 'b')
plt.plot(data2, 'r')
plt.legend(['Train', 'Val'], loc='upper left')
plt.title("accuracy")

plt.subplot(1,2, 2)
plt.plot(data3, 'b')
plt.plot(data4,'r')
plt.legend(['Train', 'Val'], loc='upper left')
plt.title("loss")

plt.savefig('./img/rse_refine.png',dpi=256)

# plt.show()

def create_confusion_matrix(y_true, y_pred):
    """ creates and plots a confusion matrix given two list (targets and predictions)
    :param list y_true: list of all targets (in this case integers bc. they are indices)
    :param list y_pred: list of all predictions (in this case one-hot encoded)
    :param dict classes: a dictionary of the countries with they index representation
    """
    plain=['0', '1', '2', '3', '4', '5']
    amount_classes = 6
    print(len(y_true),len(y_true[0]))
    confusion_matrix = np.zeros((amount_classes, amount_classes),dtype=int)
    for i in range(len(y_true)):
        for idx in range(len(y_true[i])):
            target=y_true[i][idx]
            prediction=y_pred[i][idx]
            # print(target,prediction)
            confusion_matrix[target][prediction] += 1
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in plain],
                  columns = [i for i in plain])
    # print(df_cm)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,fmt='d')
    plt.savefig('./img/rse_refine_conf.png',dpi=256)
    plt.show()

predictions = np.load("./npy/1651340164/predicts.npy",allow_pickle=True)
# print(predictions[:10])
print(predictions.shape )
gt =  np.load("./npy/1651340164/gt.npy",allow_pickle=True)
# print(gt[:10])
create_confusion_matrix(gt,predictions)