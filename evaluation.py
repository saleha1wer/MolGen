# perhaps include script for ROC file, etc
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_train_and_val_using_mpl(train_loss, val_loss, name=None, save=False):
    """
    Plots the train and validation loss using Matplotlib
    """
    assert len(train_loss) == len(val_loss)

    f, ax = plt.subplots()
    x = np.arange(len(train_loss))
    ax.plot(x, np.array(train_loss), label='train')
    ax.plot(x, np.array(val_loss), label='val')
    ax.legend()
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    if save:
        if name != None:
            ax.set_title(name)
        else:
            ax.set_title('Anonymous plot')
        plt.savefig(name + '.png')
    return f, ax


def threshold_affinity(values, threshold=6.5):
    new_array = []
    for value in values:
        if value >= threshold:
            new_array.append(1)
        else:
            new_array.append(0)
    return new_array

def plot_roc(y_true,y_preds,labels): 
    """
    y_true is a list of real affinity values:
    [5,6,3,8...]
    y_preds is a list of lists, len > 0. Each list in y_preds being a models predictions and the same size as the true values list:
    [y_pred_model1,y_pred_model2,y_pred_model3...]
    labels are the names of the models in the same order as the y_preds list
    """
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    org_y_true = y_true
    for idx,y_pred in enumerate(y_preds):
        y_true, y_pred = threshold_affinity(org_y_true), threshold_affinity(y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_true,  y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        plt.plot(fpr,tpr,label=labels[idx]+' auc= {}'.format(str(round(auc,3))))
    plt.legend()
    plt.savefig('images/roc_curve')
    plt.show()

