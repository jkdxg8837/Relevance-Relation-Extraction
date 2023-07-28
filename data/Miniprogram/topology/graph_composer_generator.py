'''
Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved.
'''
import networkx as nx
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sys

args = str(sys.argv)
train_path = args[0]
test_path = args[1]
output_dir = args[2]


relation_data = []

with open(train_path, 'r') as f:
    for line in f:
        service, rela, content = line.strip().split('\t')
        relation_data.append((service, rela, content))
# relations = ['明确强相关', '偏强相关', '弱相关', '不相关']
relations = ['_hypernym', '_derivationally_related_form', '_instance_hypernym', '_also_see', '_member_meronym', '_synset_domain_topic_of', \
            '_has_part', '_member_of_domain_usage', '_member_of_domain_region', '_verb_group', '_similar_to' ]

test_relations = []
with open(test_path, 'r') as f:
    for line in f:
        service, rela, content = line.strip().split('\t')
        test_relations.append((service, content))

# print(relation_data.info())

g = nx.Graph()

for row in relation_data:
    serv_id = row[0]
    serv_name = "serv_name"
    serv_desc = "serv_desc"
    serv_detail_desc = "detail_desc"

    # author = str(row['author'])
    

    cont_id = row[2]
    cont_title = "cont_title"

    relation = relations.index(row[1])
    # relation_weight = [1, 0.8, 0.6, 0]

    added_nodes_info = [(serv_id, {'node_type': 'service', 'serv_name': serv_name, 'serv_desc': serv_desc, 'serv_detail_desc': serv_detail_desc}),
                        (cont_id, {'node_type': 'content', 'cont_title': cont_title}),
                        ]
    

    g.add_nodes_from(added_nodes_info)

    # add relation edge
    # if relation != 3:
        # g.add_edges_from([(serv_id, cont_id, {'weight': relation_weight[relation]})])
    if relation != 6:
        g.add_edges_from([(cont_id, serv_id)])
    if relation ==3 or relation == 10 or relation == 6:
        g.add_edges_from([(serv_id, cont_id)])
    


print(g)
# remove test relations
g.remove_edges_from(test_relations)
print('After removing test:', g)

nx.write_gpickle(g, '../data/g.pkl')
nx.write_adjlist(g, '../data/g.adjlist')
print("Wrting done!")
# print(graph)

pos = nx.spring_layout(g)
# print('hi')
nx.draw_networkx(g, pos, nodelist=[node for node, nattr in g.nodes.items() if nattr['node_type'] == 'service'],
                 color='orange', alpha=.5, with_labels=False)
# nx.draw_networkx(g, pos, nodelist=[node for node, nattr in g.nodes.items() if nattr['node_type'] == 'content'],
#                  color='purple', alpha=.5, with_labels=False)
# print('hiii')
#
nx.draw_networkx()
plt.show()
plt.savefig('adj.png')
#
#
print(g)

