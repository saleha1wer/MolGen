import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.interpolate import make_interp_spline


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
def smooth(array, weight):
    for value in range(len(array) - 2):
        array[value] = array[value] * weight + (1 - weight) * array[value+1]
    return array

def shorten(array,new_length,chemprop=False):
    if chemprop:
        k = 6
    else:
        k = 2
    v = k*(len(array)-new_length)
    temp = array[-v:]
    temp = chunks(temp,k)
    array = array[:len(array)-int(v)]
    temp = [np.mean(i) for i in temp]
    array.extend(temp)
    print(len(array))
    return array[:new_length]

def get_values_from_path(path):
    fullpath = path + 'train_loss.csv'
    df = pd.read_csv(fullpath)
    train_values = df['Value']
    train_values = np.array(train_values).tolist()

    fullpath = path + 'val_loss.csv'
    df = pd.read_csv(fullpath)
    val_values = df['Value']
    val_values = np.array(val_values).tolist()
    val_values = df['Value']
    
    return train_values,val_values




paths = ['training/GAT/GAT-',
        'training/GIN/GIN-','training/chemprop/chemprop-']

# labels = ['ChemProp','GAT','GIN']
labels = ['GAT-1','GIN-1','ChemProp']
smth = 0.4

line_effects = {
    'training/GAT/GAT-' :{'color':'darkgoldenrod',
                        'path_effects': [path_effects.SimpleLineShadow(offset=(0, 0), shadow_color='darkgoldenrod', alpha=0.5, rho=0.3,linewidth=3),path_effects.Normal()]
    },
    'training/GIN/GIN-' :{'color':'royalblue',
                        'path_effects': [path_effects.SimpleLineShadow(offset=(0, 0), shadow_color='royalblue', alpha=0.5, rho=0.3,linewidth=3),path_effects.Normal()]},           
    'training/chemprop/chemprop-': {'color':'black',
                        'path_effects': [path_effects.SimpleLineShadow(offset=(0, 0), shadow_color='black', alpha=0.5, rho=0.3,linewidth=3),path_effects.Normal()]}
}
train_style= '-'
val_style= (0, (5, 5))
train_size = 1
val_size = 1
plt.figure(figsize=(10.9, 7))

for path in paths:
    train_values,val_values = get_values_from_path(path)
    if 'chemprop' in path:
        train_values = shorten(train_values,100,chemprop=True)
    else:
        train_values = shorten(train_values,150)
        train_values = smooth(train_values,smth)
        val_values = smooth(val_values,smth)
    if 'chemprop' in path:
        val_values =  np.array(val_values) **2
        val_values = val_values.tolist()
    
    plt.plot(train_values,linestyle=train_style, linewidth=train_size,label=labels[paths.index(path)],**line_effects[path])
    plt.plot(val_values,linestyle=val_style, linewidth=val_size,**line_effects[path])

# plt.figure(figsize=(10.9, 7))
# for idx,valid in enumerate(all_valids):
#     valid = smooth(valid,smth)
#     plt.plot(valid,label=labels[idx]+' validation')
#     # plt.plot(valid,label=labels[idx]+' validation loss')

# for idx,train in enumerate(all_trains):
#     train = np.array(train)
#     train = smooth(train,smth)
#     plt.plot(train,label=labels[idx]+' train')
    # plt.plot(train,label=labels[idx]+' train loss')
plt.ylim(0,1.3)
plt.xlim(0,150)
plt.legend(prop={'size': 20})
plt.ylabel('Loss',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.axhspan(ymin=0,ymax=2,xmin=0,xmax=150,facecolor='grey',alpha=0.2)

plt.savefig('training.png')

plt.show()








