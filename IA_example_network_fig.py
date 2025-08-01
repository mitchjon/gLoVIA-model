import scaling_generalized_lotka_volterra_ia_5h as gLV_IA
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as panda


on=0.9
off=0.01

lgrowth=0.3
wgrowth=0.3

wordlist = ['cat', 'car', 'bar']#,'beg']


Model1 = gLV_IA.Model(wordlist, node_sort = 'alphabetical', seed = 58)
Model1.set_r_method('method1',lgrowth = 0.3, wgrowth = 0.3)
Model1.create_pattern_matrix(10, 0)
Model1.set_xZero_method('letter activation', on = 0.9, off = 0.01, sigma = 0, seed = 23)
Model1.set_M_method('method4', ldecay_in=-1,wdecay_in=-1,ltl_in=-0.9,wtw_in=-0.5)
#Model1.set_forcing_method('no forcing')
#Model1.set_simulator('stochastic moments u substitution 3rd order central moment neglect', sigma = 0.3)
Model1.set_stop_condition('accumulated gap')

Model1.initialize_all_params()
w_matrix = Model1.interaction_matrix
print(w_matrix)

#G = nx.from_numpy_array(w_matrix, create_using = nx.DiGraph)

#nx.set_node_attributes(G,Model1.nodelist, name = 'labels')
#print(G.nodes)




letters = Model1.letterlist
words = Model1.wordlist
nodes = Model1.nodelist

#w_matrix = Model1.pattern_matrix
colormap = mpl.colormaps['seismic']
fig, ax = plt.subplots()
ax.xaxis.tick_top()
plt.imshow(w_matrix, cmap = colormap, vmin=-1,vmax=1);plt.xticks(np.arange(len(nodes)),nodes);plt.yticks(np.arange(len(nodes)),nodes);plt.colorbar()

plt.savefig(r"C:\Users\mitch\Documents\OSU\Research data and code\gLoVIA-paper\example_IA_adjacency_mtx.png", transparent=True)


plt.show()


plt.imshow(Model1.pattern_matrix, cmap = colormap, vmin=-1,vmax=1);plt.xticks(np.arange(len(words)),words);plt.yticks(np.arange(len(nodes)),nodes);plt.colorbar()
plt.show()
print(Model1.nodelist)
print(w_matrix)




edges = [( nodes[i],nodes[j],w_matrix[i,j] ) for i in range(Model1.n_nodes) for j in range(Model1.n_nodes)]
print(edges)
#print(edges[7][0] == edges[7][1])
#edges.pop(0)
#print(edges)

filtered_edges = []
for edge in edges:
    if edge[0]!=edge[1] and  np.absolute(edge[2]) > 0.2:
        newedge = (edge[0], edge[1], edge[2])
        filtered_edges.append(newedge)

print(filtered_edges)

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(filtered_edges, weight = 'weight')
print(G.nodes)
print(G.edges)

#letter_pos = [[0.1,(i+1)*0.1] for i in range(len(letters))]
#word_pos =   [[0.9,(i+1)*0.1] for i in range(len(words))]



letter_xpos = [pos - np.mean(list(range(len(letters)))) for pos in range(len(letters))]
circle_x_to_y = lambda r,h,k,x: [   k+np.sqrt(r**2-(x-h)**2)  , k-np.sqrt(r**2-(x-h)**2)    ] [np.argmin(  [   np.absolute(k+np.sqrt(r**2-(x-h)**2))  , np.absolute(k-np.sqrt(r**2-(x-h)**2))    ]  )]
letter_ypos = [circle_x_to_y(13,0,-14,x) for x in letter_xpos]

word_xpos =   [pos - np.mean(list(range(len(words))))   for pos in range(len(words))]
word_ypos = [circle_x_to_y(1.4,0,0,x) for x in word_xpos]


letter_pos = list(zip(letter_xpos,letter_ypos))
word_pos =   list(zip(word_xpos,word_ypos))

nodes = letters + words
node_pos = letter_pos + word_pos

pos_dict = dict(zip(nodes, node_pos))
print(pos_dict)

#df = panda.DataFrame({'from':['R','R','D04','D04','D06','D06'], 'to':['D04','D06','R','D06','R','D04']})
#G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
#G['R']['D04']['weight'] = 243.0
#G['R']['D06']['weight'] = 150.0
#G['D06']['D04']['weight'] = 211.0
#pos = nx.spring_layout(G)
#instead of spring, set the positions yourself
#labelPosDict = {'R':[0.1,0.1], 'D04':[0.5,.9], 'D06':[.9,.18]}
labels = nx.get_edge_attributes(G,'weight')
#print(labels)

#nx.draw_networkx_edge_labels(G,pos=pos_dict)
#plt.axvline(.1, alpha=0.1, color='green')
#plt.axhline(.3, alpha=0.1, color='green')
#Create a dict of fixed node positions
#nodePosDict = {'R':[0.1,0.3], 'D04':[0.5,.9], 'D06':[.9,.18]}
# Make the graph - add the pos and connectionstyle arguments
weights = [e[2] for e in filtered_edges]
#minw = np.amin(weights)
#weights = [e-minw +0.01 for e in weights]
#maxw = np.amax(weights)
#weights = [(e*255/maxw)//1 for e in weights]
colormap = mpl.colormaps['seismic']
print('-----------------------------',  colormap(0.4), [colormap((i/5)-5)[2] for i in range(10)])

#plt.imshow(np.repeat(np.expand_dims(np.array([colormap(i) for i in range(255)]),axis=1),100,axis=1) )
#plt.show()
#print(weights)

#nx.draw_networkx_nodes(G, pos_dict, nodelist=nodes, node_size=1500, node_color='grey', node_shape='o', alpha=0.3, linewidths=None,  label=nodes, margins = )


#ConSty = ConnectionStyle("Arc3, rad=0.2")
#ConSty.Angle3
print(len(weights), len(G.edges))

#nx.draw(G, pos = pos_dict, with_labels = True,
#        node_size=1500, alpha=0.7, font_weight="bold", arrows=True,
#       connectionstyle='arc3, rad = 0.1', edge_cmap = colormap, edge_color = weights,style = 'solid', linewidths = 1, width = 1, arrowstyle = 'simple')
#plt.axis('on')


nx.draw_networkx_nodes(G, pos_dict, nodelist=nodes, node_size=1300, node_color='#1f78b4', node_shape='o', alpha=0.7, cmap=None, vmin=None, vmax=None, ax=None, linewidths=1, edgecolors=None, label=None, margins=None)
nx.draw_networkx_edges(G, pos_dict, edgelist=filtered_edges, width=1, edge_color=weights, style='solid', alpha=0.8, arrowstyle='simple', arrowsize=15, edge_cmap=colormap, edge_vmin=None, edge_vmax=None, ax=None, arrows=None, label=None, node_size=1300, nodelist=None, node_shape='o', connectionstyle='arc3, rad = 0.2', min_source_margin=0, min_target_margin=0)
nx.draw_networkx_labels(G, pos_dict, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, clip_on=True)


plt.axis('on')

plt.savefig(r"C:\Users\mitch\Documents\OSU\Research data and code\gLoVIA-paper\example_IA_graph.png", transparent=True)

plt.show()












#pos=networkx.spring_layout(G)
#for i in range(len(P)):
#    networkx.draw_networkx_edges(G,pos,
#                edgelist=[list(P[i]), list(Q[perms[0][i]]), list(R[perms[1][i]])],edge_color=color_map[i], width="8")
