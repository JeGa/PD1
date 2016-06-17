import numpy as np
import logging
import networkx
import os
from scipy import misc


def loadunaryfile(filename):
    file = open(filename, "r")

    xsize = int(file.readline())
    ysize = int(file.readline())
    labels = int(file.readline())

    data = np.empty((ysize, xsize, labels))

    for x in range(xsize):
        for y in range(ysize):
            for l in range(labels):
                data[y, x, l] = float(file.readline())

    return data


def readimg(imagename):
    img = misc.imread(os.path.join("data", imagename))
    img = np.array(img, dtype=np.float64) / 255
    return img


class Node:
    def __init__(self, y, x):
        self.y = y
        self.x = x

    def pos(self):
        return self.y, self.x


class Nodegrid:
    def __init__(self, ysize, xsize):
        # Create grid of nodes
        self.nodegrid = [[Node(y, x) for x in range(xsize)] for y in range(ysize)]

        self.g = networkx.DiGraph()
        for nodelist in self.nodegrid:
            self.g.add_nodes_from(nodelist)

        # Source node
        self.source = Node(-1, -1)
        self.sink = Node(-1, -1)

        self.g.add_node(self.source)
        self.g.add_node(self.sink)

        self.ysize = ysize
        self.xsize = xsize

    def loop(self, edgecallback, nodecallback):
        """
        Loops over the grid of nodes. Two callback functions are required:

        :param edgecallback: Called for every edge.
        :param nodecallback: Called for every node.
        """
        logging.info("Iterate through graph.")

        for y in range(self.ysize - 1):
            for x in range(self.xsize - 1):
                node_i = self.nodegrid[y][x]

                # Node
                nodecallback(node_i)

                # Right edge
                node_j = self.nodegrid[y][x + 1]
                edgecallback(node_i, node_j)

                # Down edge
                node_j = self.nodegrid[y + 1][x]
                edgecallback(node_i, node_j)

        # Last column
        for y in range(self.ysize - 1):
            node_i = self.nodegrid[y][self.xsize - 1]

            # Node
            nodecallback(node_i)

            # Down edge
            node_j = self.nodegrid[y + 1][self.xsize - 1]
            edgecallback(node_i, node_j)

        # Last row
        for x in range(self.xsize - 1):
            node_i = self.nodegrid[self.ysize - 1][x]

            # Node
            nodecallback(node_i)

            # Right edge
            node_j = self.nodegrid[self.ysize - 1][x + 1]
            edgecallback(node_i, node_j)

        # Last node
        nodecallback(self.nodegrid[self.ysize - 1][self.xsize - 1])

    def loopedges(self, callback):
        logging.info("Iterate through edges.")

        for y in range(self.ysize - 1):
            for x in range(self.xsize - 1):
                node_i = self.nodegrid[y][x]

                # Right edge
                node_j = self.nodegrid[y][x + 1]
                callback(node_i, node_j)

                # Down edge
                node_j = self.nodegrid[y + 1][x]
                callback(node_i, node_j)

        # Last column
        for y in range(self.ysize - 1):
            node_i = self.nodegrid[y][self.xsize - 1]

            # Down edge
            node_j = self.nodegrid[y + 1][self.xsize - 1]
            callback(node_i, node_j)

        # Last row
        for x in range(self.xsize - 1):
            node_i = self.nodegrid[self.ysize - 1][x]

            # Right edge
            node_j = self.nodegrid[self.ysize - 1][x + 1]
            callback(node_i, node_j)

    def loopnodes(self, callback):
        logging.info("Iterate through nodes.")
        for y in range(self.ysize):
            for x in range(self.xsize):
                callback(self.nodegrid[y][x])

    def add_edge(self, node_i, node_j, capacity):
        self.g.add_edge(node_i, node_j, capacity=capacity)

    def add_source_edge(self, node, capacity):
        self.g.add_edge(self.source, node, capacity=capacity)

    def add_sink_edge(self, node, capacity):
        self.g.add_edge(node, self.sink, capacity=capacity)

    def maxflow(self):
        logging.info("Calculate max flow.")
        _, partition = networkx.minimum_cut(self.g, self.source, self.sink)
        return partition

    def draw(self):
        positions = {}
        for nodelist in self.nodegrid:
            for node in nodelist:
                positions[node] = [node.x, node.y]

        pad = 2
        nodesize = 10
        positions[self.source] = [self.xsize / 2 - 0.5, -pad]
        positions[self.sink] = [self.xsize / 2 - 0.5, self.ysize + pad]
        networkx.draw_networkx(self.g, pos=positions,
                               node_size=nodesize, with_labels=False,
                               width=0.5)
