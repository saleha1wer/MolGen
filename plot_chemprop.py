from bitarray import test
import matplotlib.pyplot as plt
import pandas as pd
hidden_sizes = [1,5,10,20,50,100,200,350,500,750,1000,1500,2000,3000,4000,5000,6000,7500]
results = []
for size in hidden_sizes:
    path = 'chemprop_hiddensize/'+str(size)+'/test_scores.csv'
    test_score = pd.read_csv(path)
    test_score = test_score['Mean rmse']
    results.append(test_score[0])

plt.plot(hidden_sizes,results)
plt.xlabel('Hidden Size')
plt.ylabel('Mean RMSE')
plt.show()