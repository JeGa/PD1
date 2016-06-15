import numpy as np
import logging
import networkx


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


class Node:
    def __init__(self, y, x):
        self.y = y
        self.x = x


class Nodegrid:
    def __init__(self, ysize, xsize):
        # Create grid of nodes
        self.nodegrid = [[Node(y, x) for x in range(xsize)] for y in range(ysize)]

        self.g = networkx.DiGraph()
        for nodelist in self.nodegrid:
            self.g.add_nodes_from(nodelist)

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
                nodecallback(node_i, self.g)

                # Right edge
                node_j = self.nodegrid[y][x + 1]
                edgecallback(node_i, node_j, self.g)

                # Down edge
                node_j = self.nodegrid[y + 1][x]
                edgecallback(node_i, node_j, self.g)

        # Last column
        for y in range(self.ysize - 1):
            node_i = self.nodegrid[y][self.xsize - 1]

            # Node
            nodecallback(node_i, self.g)

            # Down edge
            node_j = self.nodegrid[y + 1][self.xsize - 1]
            edgecallback(node_i, node_j, self.g)

        # Last row
        for x in range(self.xsize - 1):
            node_i = self.nodegrid[self.ysize - 1][x]

            # Node
            nodecallback(node_i, self.g)

            # Right edge
            node_j = self.nodegrid[self.ysize - 1][x + 1]
            edgecallback(node_i, node_j, self.g)

        # Last node
        nodecallback(self.nodegrid[self.ysize - 1][self.xsize - 1], self.g)

    def loopnodes(self, callback):
        logging.info("Iterate through nodes.")
        for y in range(self.ysize):
            for x in range(self.xsize):
                callback(self.nodegrid[y][x], self.g)

    def maxflow(self):
        logging.info("Calculate max flow.")
        _, partition = self.g.minimum_cut()
        return partition

    def draw(self):
        positions = {}
        for nodelist in self.nodegrid:
            for node in nodelist:
                positions[node] = [node.x, node.y]
        networkx.draw_networkx(self.g, pos=positions, node_size=10, with_labels=False)
