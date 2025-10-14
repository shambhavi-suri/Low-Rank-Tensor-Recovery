import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

itr = 150
m_choices = [0.03, 0.04, 0.08, 0.12] 
data_array = np.empty((8, itr, 3, len(m_choices)))
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

for i in range(0,8):
    data_array[i,:,0,:] = np.array(pd.read_csv(f"data_atiht_{i}.csv", index_col=0))
    #data_array[i,:,1,:] = np.array(pd.read_csv(f"data_kztiht_{i}.csv", index_col=0))
    #data_array[i,:,2,:] = np.array(pd.read_csv(f"data_tiht_{i}.csv", index_col=0))

### Getting plot data from this 

median_data = np.zeros((itr, len(m_choices)))
qlow_data = np.zeros((itr, len(m_choices)))
qup_data = np.zeros((itr, len(m_choices)))
median_data_kz = np.zeros((itr, len(m_choices)))
qlow_data_kz = np.zeros((itr, len(m_choices)))
qup_data_kz = np.zeros((itr, len(m_choices)))
median_data_iht = np.zeros((itr, len(m_choices)))
qlow_data_iht = np.zeros((itr, len(m_choices)))
qup_data_iht = np.zeros((itr, len(m_choices)))

for i in range(len(m_choices)):

    median_data[:, i] = np.median(data_array[:,:, 0, i],axis = 0)
    qlow_data[:,i] = np.quantile(data_array[:,:, 0, i],q = 0.25, axis = 0)
    qup_data[:,i] = np.quantile(data_array[:,:, 0, i],q = 0.75, axis = 0)
    median_data_kz[:,i] = np.median(data_array[:,:, 1, i],axis = 0)
    qlow_data_kz[:,i] = np.quantile(data_array[:,:, 1, i], q = 0.25, axis = 0)
    qup_data_kz[:,i] = np.quantile(data_array[:,:, 1, i], q = 0.75, axis = 0)
    median_data_kz[:,i] = np.median(data_array[:,:, 2, i],axis = 0)
    qlow_data_iht[:,i] = np.quantile(data_array[:,:, 2, i], q = 0.25, axis = 0)
    qup_data_iht[:,i] = np.quantile(data_array[:,:, 2, i], q = 0.75, axis = 0)

from matplotlib.lines import Line2D
fig,ax = plt.subplots(1,1,figsize=(6,4))
linewidth = [2,2,2,2]
linestyle = ['-','dotted','dashed']

from matplotlib.lines import Line2D
fig,ax = plt.subplots(1,1,figsize=(6,4))
linewidth = [2,2,2,2]
linestyle = ['-','dotted','dashed','-.']

for i in range(len(m_choices)):
    mean = median_data[:,i]
    qlow = qlow_data[:,i]
    qhigh = qup_data[:,i]
    plt.plot(range(itr),mean, color  = colors[i], label = "TrimTIHT m="+str(m_choices[i]), linestyle = linestyle[i], marker = 'P',
            markevery = 7, linewidth = linewidth[i])
    plt.fill_between(range(itr),qhigh,qlow,alpha=0.2, color  = colors[i])

'''i
for i in range(len(m_choices)):
    mean = median_data_kz[:,i]
    qlow = qlow_data_kz[:,i]
    qhigh = qup_data_kz[:,i]
    plt.plot(range(itr),mean, color  = colors[i], label = "KaczTIHT m="+str(m_choices[i]),linestyle = linestyle[i],
            marker = 's', markevery = 6, linewidth = linewidth[i])
    plt.fill_between(range(itr),qlow,qhigh,alpha=0.2, color  = colors[i])

for i in range(len(m_choices)):
    mean = median_data_iht[:,i]
    qlow = qlow_data_iht[:,i]
    qhigh = qup_data_iht[:,i]
    plt.plot(range(itr),mean, color  = colors[i], label = "TIHT m="+str(m_choices[i]),linestyle = linestyle[i],
            marker = 'o', markevery = 6, linewidth = linewidth[i])
    plt.fill_between(range(itr),qlow,qhigh,alpha=0.2, color  = colors[i])
'''
custom_lines = [Line2D([0], [0], color=colors[0], linewidth = 2, linestyle = '-'),
        Line2D([0], [0], color=colors[1],linewidth = 2,linestyle = 'dotted'),
        Line2D([0], [0], color=colors[2],linewidth =2, linestyle = 'dashed'),
        Line2D([0], [0], color=colors[3],linewidth =2, linestyle = '-.')]
legend_1 = plt.legend(custom_lines, [r'$\kappa$' +  f'$ = {m_choices[0]}$',
            r'$\kappa$' +  f'$ = {m_choices[1]}$',r'$\kappa$' +  f'$= {m_choices[2]}$',
            r'$\kappa$' +  f'$= {m_choices[3]}$'],
            fontsize='medium', loc = 'lower left')

#custom_lines_2 = [Line2D([0], [0], color='black',marker = 's'),Line2D([0], [0], color='black',marker = 'P'),Line2D([0], [0], color='black',marker = 'o')]
plt.gca().add_artist(legend_1)

#plt.legend(custom_lines_2, ['KaczTIHT','TrimTIHT', 'TIHT'], fontsize='medium', loc = 'upper right')

plt.yscale("log")
#plt.ylim([1e-6, 10**2])

plt.xlabel(xlabel=r'Iterations',labelpad = 0, fontsize = '12')
plt.ylabel(ylabel=r'Relative Error',labelpad = 0, fontsize = '12')
plt.show()
#plt.title("Face-Splitting Measurements")
fig.savefig('fsTrimTIHTvsKaczTIHTcomp_3d.png', dpi=300,  bbox_inches="tight")
