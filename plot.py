import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
from scipy.stats import pearsonr as pr

targets = [line.rstrip() for line in open(sys.argv[1])]

alldata = {'TARGET':[], 'MODEL':[], 'TM':[], 'QA':[]}

for target in targets: 
    tmdic = {}
    for line in open('model_tmscores/{}_tmlist'.format(target)):
        tmdic[line.split()[0]] = float(line.split()[1].rstrip())

    qadic = {}
    for line in open('proQ4_QA/{}.3D.srv.ta'.format(target)):
        qadic[line.split()[0]] = float(line.split()[1].rstrip())

    for key in tmdic:
        alldata['TARGET'].append(target)
        alldata['MODEL'].append(key)
        alldata['TM'].append(tmdic[key])
        alldata['QA'].append(qadic[key])


df = pd.DataFrame(alldata)

fig, axs = plt.subplots(3, 6, sharex=True, sharey=True)
plt.xticks(np.arange(0, 1.2, 0.2))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlim(0,1)
plt.ylim(0,0.7)

row = col = 0
for target in targets:
    tdf = df.loc[df['TARGET']==target]
    m, b = np.polyfit(list(tdf['TM']), list(tdf['QA']), 1)
    pcc = pr(list(tdf['TM']), list(tdf['QA']))
    x = np.arange(0,1,0.01)
    axs[row][col].plot(x, m*x + b)
    sb.scatterplot(x='TM', y='QA', data=tdf, s=5, ax=axs[row][col])
    axs[row][col].set_ylabel('')
    axs[row][col].set_xlabel('')
    axs[row][col].set_title('{t} - PCC:{p}'.format(t=target, p=round(pcc[0], 3)), fontsize=8)

    if col < 5: col += 1
    else: 
        col = 0
        row += 1


fig.text(0.5, 0.04, 'TM score', ha='center', fontsize=12)
fig.text(0.08, 0.5, 'ProQ4 score', va='center', rotation='vertical', fontsize=12)
plt.show()



