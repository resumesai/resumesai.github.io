import pandas as pd
import graphviz
import math
import pydotplus
import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from IPython.display import Image


def make_tree(a):
    b = pd.read_csv("binary.csv")
    target = pd.read_csv("answers.csv")
    targets = []
    for j in range(len(target)):
        targets.append([target[i][j] for i in target][1:])
    target_labels = list(target[:])[1:]
    binary_labels = list(b["filename"])
    features = []
    for i in target_labels:
        k = 0
        for j in binary_labels:
            if (i.split("_")[0]) == j:
                features.append([b[l][k] for l in b][1:])
            k += 1
    features = [f[1:] for f in features]
    features = [[-1 if i > 0 else 1 for i in f] for f in features]
    A_NUMBER = "A20349472"
    random_state_2 = int(A_NUMBER[6:9])
    binary_targets = [-1 if t > 0 else 1 for t in targets[a]]
    model = DecisionTreeClassifier(max_depth=3, random_state=random_state_2)
    result = model.fit(features, binary_targets)
    dot_data = StringIO()
    columns = list(b[1:])[2:]
    tree.export_graphviz(
        model,
        out_file=dot_data,
        feature_names=columns,
        class_names=["accepted", "rejected"],
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    leaves = set()
    non_leaves = set()
    for edge in graph.get_edges():
        leaves.add(int(edge.get_destination()))
        non_leaves.add(int(edge.get_source()))
    leaves.difference_update(non_leaves)
    leaves_idx = list(filter(lambda l: type(l) is int, leaves))
    nodes = graph.get_nodes()
    terminal_nodes = [nodes[i + 1] for i in leaves_idx]
    for node in terminal_nodes:
        text = node.get_label()
        if text:
            cls = text[1:-1].split("nclass = ")[1]
            gini = float(text[1:-1].split("gini = ")[1].split("\\nsamples")[0])
            if gini < 0.5:
                node.set("style", "filled")
                if cls == "accepted":
                    node.set("fillcolor", "#64CD6D")
                elif cls == "rejected":
                    node.set("fillcolor", "#EB4C63")
    for node in nodes:
        text = node.get_label()
        if text:
            sin = "<= 0.0"
            if sin in text:
                newtext = "".join(text.split(sin))
                node.set_label(newtext)
    graph.write_png("trees/rater{}.png".format(a))
    return "analysis/trees/rater{}.png".format(a)
    # return Image(graph.create_png())
