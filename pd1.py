"""
Solve the metric labeling problem: Minimize the following energy function.

E(y) = SUM_nodes(unary(yi, xi)) + SUM_edges(pairwise(yi, yj, xi, xj))

The aim is to always ensure for all feasible primal dual solutions the
relaxed complementary primal slackness condition.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import os

import utility


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

        # These variables change for each iteration.
        self.currentLabel = None
        self.currentGraph = None

    def initPrimals(self):
        return np.random.randint(0, self.numlabels, (self.ysize, self.xsize))

    def initDuals(self):
        # TODO
        pass

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

    def h(self):
        # TODO
        pass

    def precompdmin(self):
        """
        Compute the minimum of all distances between all neighboring pixel.
        Since the Potts model is used it suffices to just use two distinct labels.
        (Not all label combinations need to be computed)

        :return: dmin.
        """

        # Using the graph class since it already provides edge traversal.
        # Creating a temporary dummy graph.
        dummy = utility.Nodegrid(self.ysize, self.xsize)

        # Initialize to first edge distance.
        dmin = self.d(1, 0, self.img[0, 0], self.img[0, 1])

        def edge(node_i, node_j):
            nonlocal dmin

            temp = self.d(1, 0,
                          self.img[node_i.pos()],
                          self.img[node_j.pos()])

            if temp < dmin:
                dmin = temp

        dummy.loopedges(edge)

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

            # TODO: Balance variable
            cap = ((self.w * self.dmin) / 2) - 1

            # cap_pq
            self.currentGraph.add_edge(node_i, node_j, cap)

            # cap_qp
            self.currentGraph.add_edge(node_j, node_i, cap)

    def nodecallback(self, node_i):
        # Height of vertex (active label)
        hxp = self.h(node_i)
        # Height of label c (current label in iteration)
        hc = self.h(node_i, self.currentLabel)

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

    def update_duals_primals(self, c):
        """
        Update the duals having currently label c.
        Construct a graph and rearrange the heights of the duals.

        :return:
        """
        pass

    def segment(self):
        for c in self.labels:
            self.update_duals_primals()


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

    w = 1
    l = 0.5
    pd1 = PD1(img, unaries, numlabels, w, l)

    logging.info("Save image.")
    plt.imshow(img)
    plt.show()
    plt.imsave("img_out", img)


if __name__ == '__main__':
    main()
