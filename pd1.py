"""
Solve the metric labeling problem: Minimize the following energy function.

E(y) = SUM_nodes(unary(yi, xi)) + SUM_edges(pairwise(yi, yj, xi, xj))

The aim is to always ensure for all feasible primal dual solutions the
relaxed complementary primal slackness condition.

TODO:
- Dmin for each pixel or for all?
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import random

import utility


class Duals:
    """
    Saves the duals and balance variables.
    - For each node there is one dual variable (h_xp).
    - For each edge there are two balance variables for each label.

    Only save one direction ypq,a since ypq,a = -yqp,a.
    The edges per node are right and down (except the far right and bottom ones).

    index:
    - 0: right
    - 1: down

    The balance variables and duals could also be saved in the
    networkx graph. However, it is nicer to have them outside in
    a separate class.
    """

    def __init__(self, ysize, xsize, numlabels):
        self.ysize = ysize
        self.xsize = xsize

        self.duals = np.zeros((self.ysize, self.xsize))

        self.balance = np.zeros((self.ysize, self.xsize, 2, numlabels))

    def setdual(self, pos_i, value):
        self.duals[pos_i] = value

    def getdual(self, pos_i):
        return self.duals[pos_i]

    def setbalance(self, pos_i, pos_j, label, value):
        if self._right(pos_i, pos_j):
            self.balance[pos_i[0], pos_i[1], 0, label] = value
        elif self._down(pos_i, pos_j):
            self.balance[pos_i[0], pos_i[1], 1, label] = value
        else:
            raise IndexError("Error setting balance variable.")

    def getbalance(self, pos_i, pos_j, label):
        if self._right(pos_i, pos_j):
            return self.balance[pos_i[0], pos_i[1], 0, label]
        if self._down(pos_i, pos_j):
            return self.balance[pos_i[0], pos_i[1], 1, label]

        if self._left(pos_i, pos_j):
            return -self.balance[pos_j[0], pos_j[1], 0, label]
        if self._up(pos_i, pos_j):
            return -self.balance[pos_j[0], pos_j[1], 1, label]

        raise IndexError("Error getting balance variable.")

    def getbalanceNeighbors(self, pos_i, label):
        """
        Returns the values of the balance variables of all neighbors.
        (4-Neighborhood)

        :param pos_i: Position of the node whose neighbors should be queried for.
        :param label: Label of the balance variable.
        :return: List of the balance variables value (2 - 4 entries).
        """
        neighbors = []

        y = pos_i[0]
        x = pos_i[1]

        right = (y, x + 1)
        down = (y + 1, x)
        left = (y, x - 1)
        up = (y - 1, x)

        if self._isvalid(right):
            neighbors.append(self.getbalance(pos_i, right, label))
        if self._isvalid(down):
            neighbors.append(self.getbalance(pos_i, down, label))
        if self._isvalid(left):
            neighbors.append(self.getbalance(pos_i, left, label))
        if self._isvalid(up):
            neighbors.append(self.getbalance(pos_i, up, label))

        return neighbors

    def _right(self, pos_i, pos_j):
        if pos_i[0] == pos_j[0] and pos_i[1] == pos_j[1] - 1:
            return True
        return False

    def _left(self, pos_i, pos_j):
        if pos_i[0] == pos_j[0] and pos_i[1] == pos_j[1] + 1:
            return True
        return False

    def _up(self, pos_i, pos_j):
        if pos_i[1] == pos_j[1] and pos_i[0] == pos_j[0] + 1:
            return True
        return False

    def _down(self, pos_i, pos_j):
        if pos_i[1] == pos_j[1] and pos_i[0] == pos_j[0] - 1:
            return True
        return False

    def _isvalid(self, pos):
        y, x = pos

        if y < 0 or x < 0 or y > self.ysize - 1 or x > self.xsize:
            return False
        return True


class PD1:
    def __init__(self, img, unaries, numlabels, w, l):
        """
        Edge weights are added using callback functions.

        :param img: Numpy array with rgb image data.
        :param unaries: Numpy array with unaries for each pixel and label.
        :param numlabels: Number of lables.
        :param w: Weighting parameter for pairwise terms.
        """
        self.img = img
        self.unaries = unaries
        self.numlabels = numlabels

        self.labels = range(numlabels)
        self.ysize = img.shape[0]
        self.xsize = img.shape[1]

        self.w = w
        self.l = l
        self.dmin = self.precompdmin()

        logging.info("Dmin = " + str(self.dmin))

        # Primal variables (initial random label assignment)
        self.primals = self.initPrimals()

        # Dual variables
        self.initDuals()

        # These variables change for each iteration.
        self.currentLabel = None
        self.currentGraph = None

    def initPrimals(self):
        logging.info("Initialize primals.")
        return np.random.randint(0, self.numlabels, (self.ysize, self.xsize))

    def initDuals(self):
        logging.info("initialize balance variables.")

        self.duals = Duals(self.ysize, self.xsize, self.numlabels)

        duals = self.duals
        w = self.w
        dmin = self.dmin

        def edge(pos_i, pos_j):
            nonlocal duals, w, dmin

            activelabel_i = self.primals[pos_i]
            activelabel_j = self.primals[pos_j]

            if activelabel_i != activelabel_j:
                value = w * dmin / 2
                duals.setbalance(pos_i, pos_j, activelabel_i, value)
                duals.setbalance(pos_i, pos_j, activelabel_j, -value)

        utility.Nodegrid.loopedges_raw(edge, self.ysize, self.xsize)

        logging.info("Initialize duals.")

        def node(pos_i):
            nonlocal duals

            value = self.getminheight(pos_i)
            duals.setdual(pos_i, value)

        utility.Nodegrid.loopnodes_raw(node, self.ysize, self.xsize)

    def d(self, y1, y2, x1, x2):
        """
        Returns pairwise energy between node i and node j using the
        contrast sensitive Potts model.

        :param y1: Label of i node.
        :param y2: Label of j node.
        :param x1: Pixel value at node i.
        :param x2: Pixel value at node j.
        :return: Pairwise energy.
        """
        if y1 == y2:
            return 0.0

        # Not same label
        energy = np.exp(-self.l * np.power(np.linalg.norm(x1 - x2, 2), 2))
        return energy

    def h(self, pos_i, label):
        unary = self.unaries[pos_i[0], pos_i[1], label]

        # For all neighboring balance variables.
        neighbors = self.duals.getbalanceNeighbors(pos_i, label)

        sumofbalance = sum(neighbors)

        height = unary + sumofbalance
        return height

    def getminheight(self, pos_i):
        minheight = self.h(pos_i, self.labels[0])
        for label in range(1, self.labels[-1]):
            height = self.h(pos_i, label)

            if height < minheight:
                minheight = height

        return minheight

    def precompdmin(self):
        """
        Compute the minimum of all distances between all neighboring pixel.
        Since the Potts model is used it suffices to just use two distinct labels.
        (Not all label combinations need to be computed)

        :return: dmin.
        """
        # Initialize to first edge distance.
        dmin = self.d(1, 0, self.img[0, 0], self.img[0, 1])

        def edge(pos_i, pos_j):
            nonlocal dmin

            temp = self.d(1, 0,
                          self.img[pos_i],
                          self.img[pos_j])

            if temp < dmin:
                dmin = temp

        utility.Nodegrid.loopedges_raw(edge, self.ysize, self.xsize)

        return dmin

    def makegraph(self):
        grid = utility.Nodegrid(self.ysize, self.xsize)
        return grid

    def edgecallback(self, node_i, node_j):
        """
        Interior edges: Represent the balance variables ypq and yqp.
        Increasing the flow on ypq decreases the flow on yqp for the particular label.

        :param node_i: Starting node of the edge.
        :param node_j: End node of the edge.
        """
        if (self.primals[node_i.pos()] == self.currentLabel) \
                or (self.primals[node_j.pos()] == self.currentLabel):
            # Keep height

            # cap_pq
            self.currentGraph.add_edge(node_i, node_j, 0.0)

            # cap_qp
            self.currentGraph.add_edge(node_j, node_i, 0.0)
        else:
            # Maintain feasibility.

            cap = ((self.w * self.dmin) / 2) - self.duals.getbalance(
                node_i.pos(), node_j.pos(), self.currentLabel)

            # cap_pq
            self.currentGraph.add_edge(node_i, node_j, cap)

            # cap_qp
            self.currentGraph.add_edge(node_j, node_i, cap)

    def nodecallback(self, node_i):
        # Height of vertex (active label)
        hxp = self.h(node_i.pos(), self.primals[node_i.pos()])

        # Height of label c (current label in iteration)
        hc = self.h(node_i.pos(), self.currentLabel)

        # Case 1
        if hc < hxp:
            cap = hxp - hc
            self.currentGraph.add_source_edge(node_i, cap)

        # Case 2
        if hc >= hxp:
            cap = hc - hxp
            self.currentGraph.add_sink_edge(node_i, cap)

        # Case 3
        if self.primals[node_i.pos()] == self.currentLabel:
            self.currentGraph.add_source_edge(node_i, 1)

    def update_duals_primals(self):
        """
        Update the duals having currently label c.
        Construct a graph and rearrange the heights of the duals.
        (Holding all constraints for getting feasible solution and
        the relaxed complementary slackness)
        """
        logging.info("Update duals and primals.")

        # Set graph edges and capacities.
        self.currentGraph.loop(self.edgecallback, self.nodecallback)

        flows = self.currentGraph.maxflow()

        # Update duals based on the resulting flow on the
        # interior edges.
        def edge(node_i, node_j):
            nonlocal flows

            fpq = flows[node_i][node_j]
            fqp = flows[node_j][node_i]

            value = self.duals.getbalance(
                node_i.pos(), node_j.pos(), self.currentLabel) + fpq - fqp
            self.duals.setbalance(node_i.pos(), node_j.pos(), self.currentLabel, value)
            # self.duals.setbalance(node_j.pos(), node_i.pos(), self.currentLabel, value)

        self.currentGraph.loopedges(edge)

        # Height (based on balance variables and unary)
        # hpc = hpc + fp # s -> p
        # hpc = hpc - fp # p -> t

        # Update primals: Should the label be changed to c?
        # If there is an unsaturated path between source and node p.
        # (flow < capacity)
        def edge(node_i):
            nonlocal flows

            if self.currentGraph.hassourcepath(node_i):
                flowsource = flows[self.currentGraph.source][node_i]
                cap = self.currentGraph.getcap(node_i)

                if flowsource < cap:
                    self.primals[node_i.pos()] = self.currentLabel

        self.currentGraph.loopnodes(edge)

    def post_edit_duals(self):
        # If xp = xq = c 0> ypqc = yqpc = 0
        logging.info("Post edit duals.")

        def edge(pos_i, pos_j):
            if self.primals[pos_i] == self.currentLabel and (
                        self.primals[pos_j] == self.currentLabel):
                self.duals.setbalance(pos_i, pos_j, self.currentLabel, 0.0)
                # self.duals.setbalance(pos_j, pos_i, self.currentLabel, 0.0)

        utility.Nodegrid.loopedges_raw(edge, self.ysize, self.xsize)

        def node(pos_i):
            value = self.getminheight(pos_i)
            self.duals.setdual(pos_i, value)

        utility.Nodegrid.loopnodes_raw(node, self.ysize, self.xsize)

    def segment(self):

        # while True:
        for i in range(2):
            #oldprimals = self.primals.copy()
            for c in self.labels:
                logging.info("Label c = " + str(c))

                # Set required information for each iteration.
                self.currentLabel = c
                self.currentGraph = self.makegraph()

                self.update_duals_primals()
                self.post_edit_duals()

                # if np.array_equal(self.primals, oldprimals):
                # break

    def getLabeledImage(self):
        # Assign color.
        colors = []
        for i in self.labels:
            colors.append([random.randint(0, 255),
                           random.randint(0, 255),
                           random.randint(0, 255)])

        img = np.empty((self.ysize, self.xsize, 3))

        for y in range(self.ysize):
            for x in range(self.xsize):
                img[y, x] = colors[int(self.primals[y, x])]

        return img


def main():
    logging.basicConfig(level=logging.INFO)

    imagename = "12_33_s.bmp"
    unaryfilename = "12_33_s.c_unary.txt"

    logging.info("Read image.")
    img = utility.readimg(imagename)

    logging.info("Load unaries.")
    unaries = utility.loadunaryfile(os.path.join("data", unaryfilename))

    # Calculate energy
    unaries = -np.log(unaries)
    numlabels = unaries.shape[2]

    w = 50
    l = 0.5
    pd1 = PD1(img, unaries, numlabels, w, l)
    pd1.segment()
    img = pd1.getLabeledImage()

    logging.info("Save image.")
    plt.imshow(img)
    plt.show()
    plt.imsave("img_out", img)


if __name__ == '__main__':
    main()
