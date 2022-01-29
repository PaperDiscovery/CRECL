import matplotlib.pyplot as plt
import pickle
folder='savedata_i=3/'
with open(folder+'firstpicture_try40_temp0.13_batch64_1.pickle', 'rb') as f:
    features1_first=pickle.load(f)
with open(folder+'firstpicture_try40_temp0.13_batch64_2.pickle', 'rb') as g:
    features2_first=pickle.load(g)
with open(folder+'labelorder1.pickle', 'rb') as g:
    labelorder1=pickle.load(g)
with open(folder+'labelorder2.pickle', 'rb') as g:
    labelorder2=pickle.load(g)
with open(folder+'secondpicture_try40_temp0.13_batch64_1.pickle', 'rb') as f:
    features1_second= pickle.load(f)
with open(folder+'secondpicture_try40_temp0.13_batch64_2.pickle', 'rb') as g:
    features2_second=pickle.load(g)
colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'pink']
for i, (key, values) in enumerate(features1_first.items()):
    plt.scatter(values[:, 0], values[:, 1], c=colors[i], label='task1_' + str(key), marker='.', s=36)
for i, (key, values) in enumerate(features2_first.items()):
    plt.scatter(values[:, 0], values[:, 1], c=colors[i + len(colors) // 2], label='task2_' + str(key), marker='x', s=36)
#plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('1.png')
plt.clf()
plt.cla()
for i in range(len(labelorder1)):
    plt.scatter(features1_second[labelorder1[i]][:, 0], features1_second[labelorder1[i]][:, 1], c=colors[i], label='task1_' + str(labelorder1[i]),
                marker='.',s=36)
for i in range(len(labelorder2)):
    plt.scatter(features2_second[labelorder2[i]][:, 0], features2_second[labelorder2[i]][:, 1], c=colors[i+len(colors)//2], label='task2_' + str(labelorder2[i]),
                marker='x',s=36)
#plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('2.png')