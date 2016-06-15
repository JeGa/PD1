import numpy as np
import logging
import matplotlib.pyplot as plt
import os

import utility


class PD1:
    def __init__(self, img, unaries, numlabels, w):
        """
        Edge weights are added using callback functions.

        :param img:
        :param unaries:
        :param numlabels:
        :param w:
        :return:
        """
        self.img = img
        self.unaries = unaries
        self.numlabels = numlabels

        self.labels = range(numlabels)
        self.ysize = img.shape[0]
        self.xsize = img.shape[1]

        self.w = w
        self.dmin = self.precompdmin()

        # Label assignment (primal)
        self.assignedLabel = np.empty((self.ysize, self.xsize))
        self.currentLabel = None

    def d(self, y1, y2, x1, x2):
        """
        Returns pairwise energy between node i and node j using the Potts model.

        :param y1: Label of i node.
        :param y2: Label of j node.
        :param x1: Pixel value at node i.
        :param x2: Pixel value at node j.
        :return: Pairwise energy.
        """
        if y1 == y2:
            return 0.0

        # Not same label
        energy = self.w * np.exp(-self.l * np.power(np.linalg.norm(x1 - x2, 2), 2))
        return energy

    def precompdmin(self):
        # TODO
        pass
        # Label combinations
        # comb = []
        # for a in self.labels:
        #     for b in self.labels:
        #         if a != b:
        #             comb.append([a, b])
        #
        # distances = []
        # for y in self.ysize:
        #     for x in self.xsize:
        #         for l in comb:
        #             distances.append(self.d(l[0], l[1]))

    def makegraph(self):
        grid = utility.Nodegrid(self.ysize, self.xsize)
        return grid

    def edgecallback(self, node_i, node_j, graph):
        """
        Interior edges: Represent the balance variables ypq and yqp.
        Increasing the flow on ypq decreases the flow on yqp.
        The capacity represents the maximal allowed flow.

        :param node_i:
        :param node_j:
        :param graph:
        :return:
        """

        # Get coordinates

        yi = node_i.y
        xi = node_i.x

        yj = node_j.y
        xj = node_j.x

        if (self.assignedLabel[yi, xi] == self.currentLabel) \
                or (self.assignedLabel[yj, xj] == self.currentLabel):
            # Keep height

            # cap_pq
            graph.add_edge(node_i, node_j, capacity=0.0)

            # cap_qp
            graph.add_edge(node_j, node_i, capacity=0.0)
        else:
            # Maintain feasibility.

            # cap_pq
            graph.add_edge(node_i, node_j, capacity=0.0)

            # cap_qp
            graph.add_edge(node_j, node_i, capacity=0.0)

    def nodecallback(self, node_i, graph):
        pass

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

    # TODO

    logging.info("Save image.")
    plt.imshow(img)
    plt.show()
    plt.imsave("img_out", img)


if __name__ == '__main__':
    main()
