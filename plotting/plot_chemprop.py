import matplotlib.pyplot as plt
import pandas as pd
hidden_sizes = [50,100,200,350,500,600,700,750,800,900,1000,1250,1500,2000]
results = []
for size in hidden_sizes:
    path = 'chemprop_hiddensize/'+str(size)+'/test_scores.csv'
    test_score = pd.read_csv(path)
    test_score = test_score['Mean rmse']
    test_score = test_score **2
    results.append(test_score[0])

plt.plot(hidden_sizes,results,marker='o',color='black')
plt.xlabel('Hidden Size')
plt.ylabel('MSE on Test Set')
plt.savefig('chemprop_hiddensize.png')
plt.show()