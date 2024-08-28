import json
from matplotlib import pyplot as plt
import numpy as np

data = json.load(open('random_performance.json', 'r'))
trace_names = [el[0] for el in data]
rec = [int(el[1]) for el in data]
proc = [int(el[2]) for el in data]

ind1 = np.arange(len(data)) 
ind2 = [x + 0.35 for x in ind1]
ind3 = [x + 0.175 for x in ind1]
fig, ax = plt.subplots(figsize =(15, 15)) 
bb1 = plt.barh(ind1, proc, height=0.35, label = "Recoding Time", color='r', edgecolor ='grey')
bb2 = plt.barh(ind2, rec, height=0.35, label = "Processing Time", color='b', edgecolor ='grey')
for b in bb1:    
    ax.annotate(b.get_width(), xy=(b.get_width() + b.get_height() + 8, b.get_y()), fontsize = 16, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for b in bb2:    
    ax.annotate(b.get_width(), xy=(b.get_width() + b.get_height() + 8, b.get_y()), fontsize = 16, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
plt.legend(prop={'size': 16})
plt.yticks(ind3, trace_names, fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel('Performance', fontweight ='bold', fontsize = 20)
plt.ylabel('Random 10 recordings', fontweight ='bold', fontsize = 20)
plt.savefig("performance.png", bbox_inches='tight', dpi=300)

diffs = [round(el[2] - el[1],2) for el in data]
avg_dif = round(np.mean(diffs),2)
avg_rec = round(np.mean(rec),2)
print(diffs)
print(avg_rec, avg_dif)
per = round(avg_dif / (avg_rec / 100), 2)
print(per)
