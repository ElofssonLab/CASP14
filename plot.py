import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
from scipy.stats import pearsonr as pr

targets = [line.rstrip() for line in open(sys.argv[1])]

alldata = {'TARGET':[], 'MODEL':[], 'TM':[], 'PQA':[], 'GQA':[]}

for target in targets: 
    tmdic = {}
    for line in open('model_tmscores/{}_tmlist'.format(target)):
        tmdic[line.split()[0]] = float(line.split()[1].rstrip())

    pqadic = {}
    for line in open('proQ4/{}.stage1.3D.srv.ta_clean'.format(target)):
        pqadic[line.split()[0]] = float(line.split()[1].rstrip())
    for line in open('proQ4/{}.stage2.3D.srv.ta_clean'.format(target)):
        pqadic[line.split()[0]] = float(line.split()[1].rstrip())

    gqadic = {}
    for line in open('graphQA/{}.stage1.qa_clean'.format(target)):
        gqadic[line.split()[0]] = float(line.split()[1].rstrip())
    for line in open('graphQA/{}.stage2.qa_clean'.format(target)):
        gqadic[line.split()[0]] = float(line.split()[1].rstrip())

    for key in tmdic:
        print (key, target)
        alldata['TARGET'].append(target)
        alldata['MODEL'].append(key)
        alldata['TM'].append(tmdic[key])
        alldata['PQA'].append(pqadic[key])
        alldata['GQA'].append(gqadic[key])

df = pd.DataFrame(alldata)

fig, axs = plt.subplots(3, 6, sharex=True, sharey=True)
plt.xticks(np.arange(0, 1.2, 0.2))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlim(0,1)
plt.ylim(0,0.7)

row = col = 0
for target in targets:
    tdf = df.loc[df['TARGET']==target]
    m, b = np.polyfit(list(tdf['TM']), list(tdf['PQA']), 1)
    pcc = pr(list(tdf['TM']), list(tdf['PQA']))
    x = np.arange(0,1,0.01)
    axs[row][col].plot(x, m*x + b)
    sb.scatterplot(x='TM', y='PQA', data=tdf, s=5, ax=axs[row][col])

    m, b = np.polyfit(list(tdf['TM']), list(tdf['GQA']), 1)
    pcc = pr(list(tdf['TM']), list(tdf['GQA']))
    x = np.arange(0,1,0.01)
    axs[row][col].plot(x, m*x + b)
    sb.scatterplot(x='TM', y='GQA', data=tdf, s=5, ax=axs[row][col])

    axs[row][col].set_ylabel('')
    axs[row][col].set_xlabel('')
    axs[row][col].set_title('{t} - PCC:{p}'.format(t=target, p=round(pcc[0], 3)), fontsize=8)

    if col < 5: col += 1
    else: 
        col = 0
        row += 1

fig.text(0.5, 0.04, 'TM score', ha='center', fontsize=12)
fig.text(0.08, 0.5, 'ProQ4/GraphQA score', va='center', rotation='vertical', fontsize=12)
plt.show()



