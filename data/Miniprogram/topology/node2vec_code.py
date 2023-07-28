'''
Alipay.com Inc.
Copyright (c) 2004-2023 All Rights Reserved.
'''
import networkx as nx
from node2vec import Node2Vec
import sys
args = str(sys.argv)
output_path = args[0]

# Create a graph
graph = nx.read_gpickle(output_path + '/g.pkl')
print(graph)

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=128, walk_length=40, num_walks=20, workers=8)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
# model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use

model.wv.save_word2vec_format(output_path + '/serv2content_g_node2vec.emb')

# Save model for later use
# model.save(EMBEDDING_MODEL_FILENAME)
