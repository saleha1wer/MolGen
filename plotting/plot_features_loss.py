
import matplotlib.pyplot as plt
import numpy as np

# results = [[1, 0, 0.9348190625508627, 0.11192876831479824], [1, 1, 0.980436106522878, 0.12820097102861874], [1, 3, 0.9995683232943217, 0.25009777063709854], 
#             [4, 0, 0.6815333962440491, 0.03170938982280928], [4, 1, 0.7259259223937988, 0.05507557312348691], [4, 3, 0.6811589201291403, 0.031414029144247145], 
#             [5, 0, 0.662550151348114, 0.016294550531669626], [5, 1, 0.6894015471140543, 0.05734013680240406], [5, 3, 0.6244552334149679, 0.03013465581821229], 
#             [7, 0, 0.7241978446642557, 0.10171712565495825], [7, 1, 0.6526515682538351, 0.02167825261508942], [7, 3, 0.6589799920717875, 0.03326379410520804], 
#             [8, 0, 0.7093801697095236, 0.0233507328633697], [8, 1, 0.6511453787485758, 0.012051143645497225], [8, 3, 0.6307904124259949, 0.02147484912118789], 
#             [9, 0, 0.63877934217453, 0.015044996664697132], [9, 1, 0.6190290451049805, 0.009924046215373253], [9, 3, 0.6719421545664469, 0.04301009880539223]]
# node_to_idx = {1:0, 4:1, 5:2, 7:3, 8:4, 9:5}
# ege_to_idx = {0:0, 1:1, 3:2}

# stdev_matrix = np.zeros((3,6))
# loss_matrix = np.zeros((3,6))

# for res in results:
#     stdev_matrix[ege_to_idx[res[1]],node_to_idx[res[0]]] = float(res[3])
#     loss_matrix[ege_to_idx[res[1]],node_to_idx[res[0]]] = float(res[2])

# plt.matshow(loss_matrix, cmap='Blues')

# for i in range(loss_matrix.shape[1]):
#     for j in range(loss_matrix.shape[0]):
#         c = loss_matrix[j,i]
#         std = stdev_matrix[j,i]
#         if c != 0:
#             c = round(c,3)
#             std = round(std,3)
#             plt.text(i, j, str(c), va='center', ha='center')
#             plt.text(i, j+0.25, '({})'.format(std), va='center', ha='center')


# plt.xlabel('Number of Node Features')
# plt.ylabel('Number of Edge Features')
# plt.xticks([0,1,2,3,4,5],[1,4,5,7,8,9])
# plt.yticks([0,1,2],[0,1,3])
# cbar = plt.colorbar()
# plt.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
# plt.clim(0.6, 1) 
# # cbar.set_label('standard deviation', rotation=270) 
# plt.savefig('loss_matrix.png', bbox_inches='tight')
# plt.show()

# results = [[1, 0, 0.7663898666699728, 0.02398373815214222], 
#             [2, 0, 0.7176551024119059, 0.004527126590788488], 
#             [4, 0, 0.6780060728391012, 0.02586917524760031], 
#             [5, 0, 0.6550324757893881, 0.011712027887461306], 
#             [7, 0, 0.7078646024068197, 0.048469016095462406],
#             [8, 0, 0.7453390757242838, 0.05513706375336791], 
#             [9, 0, 0.7279788454373678, 0.019721184053860603]]

# x = [1,2,4,5,7,8,9]
# scores = []
# stds = []
# for res in results:
#     scores.append(float(res[2]))
#     stds.append(float(res[3]))
# scores = np.array(scores)
# stds = np.array(stds)
# plt.plot(x,scores,marker='o',color='red',label='Loss')
# plt.xlabel('Number of Node Features')
# plt.ylabel('Average Loss on Test Set')
# plt.fill_between(x,scores-stds,scores+stds,color='red',alpha=0.2)
# plt.savefig('loss_nodefeatures_graph_GIN.png', bbox_inches='tight')
# plt.show()


results = [[1, 0, 0.7737966966629028, 0.005851380841269714], [1, 1, 1.9966997305552165, 0.8458428910907515], [1, 3, 1.0799850622812908, 0.23353329895155794],
            [4, 0, 0.6723939776420593, 0.014281607303142959], [4, 1, 1.649145285288493, 0.7674895902435228], [4, 3, 1.2559796571731567, 0.4662269379065714],
            [5, 0, 0.6714939923286438, 0.054862826058573864], [5, 1, 4.626285592714946, 2.568025094548499], [5, 3, 0.9643628199895223, 0.14853530764787423],
            [7, 0, 0.683278341293335, 0.01532517093446624], [7, 1, 2.341879645983378, 1.8803497877345503], [7, 3, 2.9647077322006226, 2.7869392321657696],
            [8, 0, 0.7070599190394083, 0.02699289698697155], [8, 1, 3.326403776804606, 3.022620265245548], [8, 3, 1.5234358310699463, 0.3538981773444668], 
            [9, 0, 0.6964702693621317, 0.012544052122867749], [9, 1, 2.575043499469757, 2.0250634905631113], [9, 3, 0.8642287254333496, 0.06372689221079374]]

node_to_idx = {1:0, 4:1, 5:2, 7:3, 8:4, 9:5}
ege_to_idx = {0:0, 1:1, 3:2}

stdev_matrix = np.zeros((3,6))
loss_matrix = np.zeros((3,6))

for res in results:
    stdev_matrix[ege_to_idx[res[1]],node_to_idx[res[0]]] = float(res[3])
    loss_matrix[ege_to_idx[res[1]],node_to_idx[res[0]]] = float(res[2])

plt.matshow(loss_matrix, cmap='Blues')

for i in range(loss_matrix.shape[1]):
    for j in range(loss_matrix.shape[0]):
        c = loss_matrix[j,i]
        std = stdev_matrix[j,i]
        if c != 0:
            c = round(c,3)
            std = round(std,3)
            plt.text(i, j, str(c), va='center', ha='center')
            plt.text(i, j+0.25, '({})'.format(std), va='center', ha='center')
plt.xlabel('Number of Node Features')
plt.ylabel('Number of Edge Features')
plt.xticks([0,1,2,3,4,5],[1,4,5,7,8,9])
plt.yticks([0,1,2],[0,1,3])
cbar = plt.colorbar()
plt.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
plt.clim(0.6, 1) 
# cbar.set_label('standard deviation', rotation=270) 
plt.savefig('loss_matrix_GIN.png', bbox_inches='tight')
plt.show()


