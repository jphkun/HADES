#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2021 Jean-Philippe Kuntzer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------

# Authors:
# * Jean-Philippe Kuntzer

# 2D visual representation of the mesh points
#
#
#                         e3          e4
#
#
#
#                         c4                    e5
#
#     e2            c3           c5
#
#               c2
#
# e12        c12          z
#                         |
# e1        c1            0->x   c6                 e6

import numpy as np
import matplotlib.pyplot as plt
import os

from ofblockmeshdicthelper import BlockMeshDict, Vertex, SimpleGrading

# TODO: y+ precalculation
# TODO: use the y+ to set up the grading in the z direction
# TODO: add faces
# TODO: correct points repartition around the capsule
# TODO: add functions documentation


class Capsule2D:
    def __init__(self, diameter):
        self.diameter = diameter
        self.radius = diameter / 2
        # Number of point in the mesh
        self.Nx = 41
        self.Ny = 1
        self.Nz = 121
        # Expansion rate of the mesh
        self.expandX = 1/1000
        self.expandY = 1
        self.expandZ = 1
        # WARNING: Enter the correct turbulence model
        turbulenceModel = 'k-epsilon'
        self.filePath = f'../../{turbulenceModel}/2D_capsule_v2/capsule_from_prism/system/'

        # External mesh bounding box
        # (number of capsules diameters)
        self.posXmax = 4.0
        self.negXmax = -2.0
        self.posYmax = 0.1
        self.negYmax = -0.1
        self.posZmax = 4.0
        self.negZmax = 4.0

        # [deg] Points angles from center of rotation
        self.dteta12 = 170
        self.dteta2  = 155
        self.dteta4  = 75
        self.dteta45 = 60
        self.dteta5  = 45
        self.dteta56 = 23

        # x0 origin of the mesh
        self.ex0 = 0
        self.ez0 = 0

        # x1 center leading edge
        self.ex1 = -2/0.4 * self.diameter
        self.ez1 = +0.0

        # x3
        self.ex3 = 0
        self.ez3 = (self.posZmax / 0.4) * self.diameter

        # x5 center trailing edge
        self.ex6 = (self.posXmax / 0.4) * self.diameter
        self.ez6 = 0

    @staticmethod
    def circle(x0, z0, r, teta):
        """
        Computes a point on a circle at a centerain angle.
        teta goes in the trigonometric direction.
        """
        teta = np.deg2rad(teta)
        x = x0 + r * np.cos(teta)
        z = z0 + r * np.sin(teta)
        return x, z

    @staticmethod
    def splines(x0, x, teta0, R, diameter, posZmax):
        """
        Third degree, first order ODE that satisfies continuity at order 0 and
        1. This equations will be approximated by splines during the mesh
        construction
        """
        b = (3/(1*x0**2)) * \
                (R*np.sin(teta0) + x0/(3*np.tan(teta0)) - (posZmax / 0.4) *
                 diameter)
        a = (1/(3*x0**2)) * (-1/np.tan(teta0) - (2*b*x0))
        c = 0.0
        d = (posZmax / 0.4) * diameter 

        def z(x):
            return a*x**3 + b*x**2 + c*x + d

        return z(x)

    def main_points(self):
        """
        Calls functions in order to compute the main points
        """
        # Computes inlet and outler lower circle
        self.inlet_circle()
        self.outlet_circle()

    def inlet_spline(self):
        """
        Computes the points for the spline interpolation of the top part of
        the inlet.
        """
        # Points between 2 and 3
        self.ex23 = np.linspace(self.ex2, 0, 10)
        x2 = self.ex2
        teta2 = np.deg2rad(self.dteta2)
        R = (4 - self.ex1)/0.4 * self.diameter
        # Points for the spline part
        self.ez23 = self.splines(
            x2,
            self.ex23,
            teta2,
            R,
            self.diameter,
            self.posZmax)

    def inlet_circle(self):
        """
        Computes the inlet points in the circular part of the mesh
        """
        # Computes point 1
        x6 = self.ex6
        z6 = self.ez6
        r = (self.posXmax - self.ex1)/0.4 * self.diameter

        # Point 12
        teta12 = self.dteta12
        self.ex12, self.ez12 = Capsule2D.circle(x6, z6, r, teta12)

        # Point 2
        teta2 = self.dteta2
        self.ex2, self.ez2 = Capsule2D.circle(x6, z6, r, teta2)

    def outlet_spline(self):
        """
        Computes the points on the top outlet part
        """
        # Points between 3 and 4
        self.ex34 = np.linspace(self.ex4, 0, 10)

        # Constants
        x4 = self.ex4
        teta4 = np.deg2rad(self.dteta4)
        r = 4/0.4 * self.diameter

        # Points for the spline part
        self.ez34 = self.splines(
            x4,
            self.ex34,
            teta4,
            r,
            self.diameter,
            self.posZmax)

    def outlet_circle(self):
        """
        Computes the outside mesh points
        """
        # Circles related variables
        x0 = self.ex0
        z0 = self.ez0
        r  = self.posXmax/0.4 * self.diameter

        # Angles
        teta4  = self.dteta4
        teta45 = self.dteta45
        teta5  = self.dteta5
        teta56 = self.dteta56

        # Computes the points
        self.ex4,  self.ez4  = Capsule2D.circle(x0, z0, r, teta4 )
        self.ex45, self.ez45 = Capsule2D.circle(x0, z0, r, teta45)
        self.ex5,  self.ez5  = Capsule2D.circle(x0, z0, r, teta5 )
        self.ex56, self.ez56 = Capsule2D.circle(x0, z0, r, teta56)

    def bottom_points(self):
        """
        """
        # External points
        self.ex67 = +self.ex56
        self.ez67 = -self.ez56

        self.ex7 = +self.ex5
        self.ez7 = -self.ez5

        self.ex78 = +self.ex45
        self.ez78 = -self.ez45

        self.ex8 = +self.ex4
        self.ez8 = -self.ez4

        self.ex89 = +self.ex34
        self.ez89 = -self.ez34

        self.ex9 = +self.ex3
        self.ez9 = -self.ez3

        self.ex910 = +self.ex23
        self.ez910 = -self.ez23

        self.ex10 = +self.ex2
        self.ez10 = -self.ez2

        self.ex1011 = +self.ex12
        self.ez1011 = -self.ez12

        # Capsule points
        self.cx7 = +self.cx5
        self.cz7 = -self.cz5

        self.cx8 = +self.cx4
        self.cz8 = -self.cz4

        self.cx9 = +self.cx3
        self.cz9 = -self.cz3

        self.cx10 = +self.cx2
        self.cz10 = -self.cz2

        self.cx1011 = +self.cx12
        self.cz1011 = -self.cz12

    def capsule_points(self):
        """
        computes the capsules main points
        """
        ratio = self.diameter / 0.4

        # Capsule point 1 (leading edge)
        self.cx1 = -0.120 * ratio
        self.cz1 = +0.0

        # Capsule point 12, needed to construct the arc
        self.cx12 = -0.107938 * ratio
        self.cz12 = +0.068404 * ratio

        # Capsule point 2 top point of the circular part of the capsule
        self.cx2 = -0.080 * ratio
        self.cz2 = +0.120 * ratio

        # Capsule point 3 mid-point of the flat part of the capsule
        self.cx3 = -0.040 * ratio
        self.cz3 = +0.160 * ratio

        # Capsule point 4 (top), maximal diameter point
        self.cx4 = 0.0
        self.cz4 = self.diameter / 2

        # Capsule point 5 (trailing edge),
        self.cx5 = 0.080 * ratio
        self.cz5 = 0.120 * ratio

        # Capsule point 6, bottom trailing edge point
        self.cx6 = 0.080 * ratio
        self.cz6 = 0.000 * ratio

        x = [self.cx1,
             self.cx12,
             self.cx2,
             self.cx3,
             self.cx4,
             self.cx5,
             self.cx6]

        z = [self.cz1,
             self.cz12,
             self.cz2,
             self.cz3,
             self.cz4,
             self.cz5,
             self.cz6]

    def plot(self):
        """
        Point each points. Only for debugging purposes
        """
        plt.figure('Mesh')

        # External part of the mesh
        plt.plot(self.ex0, self.ez0, 'o', color='lightsteelblue', label='x0')
        plt.plot(self.ex1, self.ez1, 'x', color='lightsteelblue', label='x1')
        plt.plot(self.ex12,self.ez12,'o', color='cornflowerblue', label='x12')
        plt.plot(self.ex2, self.ez2, 'x', color='cornflowerblue', label='x2')
        plt.plot(self.ex23,self.ez23,'.', color='cornflowerblue', label='x23')
        plt.plot(self.ex3, self.ez3, 'o', color='royalblue', label='x3')
        plt.plot(self.ex34,self.ez34,'.', color='royalblue', label='x34')
        plt.plot(self.ex4, self.ez4, 'x', color='royalblue', label='x4')
        plt.plot(self.ex45,self.ez45,'o', color='blue', label='x45')
        plt.plot(self.ex5, self.ez5, 'x', color='blue', label='x5')
        plt.plot(self.ex56,self.ez56,'o', color='navy', label='x56')
        plt.plot(self.ex6, self.ez6, 'x', color='navy', label='x6')

        # Capsule mesh
        plt.plot(self.cx1, self.cz1, 'o', color='lightcoral', label='c1')
        plt.plot(self.cx12,self.cz12,'x', color='lightcoral', label='c12')
        plt.plot(self.cx2, self.cz2, 'o', color='indianred', label='c2')
        plt.plot(self.cx3, self.cz3, 'x', color='indianred', label='c3')
        plt.plot(self.cx4, self.cz4, 'o', color='firebrick', label='c4')
        plt.plot(self.cx5, self.cz5, 'x', color='firebrick', label='c5')
        plt.plot(self.cx6, self.cz6, 'o', color='red', label='c6')

        # Fancy stuff
        plt.legend()
        plt.grid()
        plt.show()

    def to_blockMesh_dict2(self):
        """
        Actual construction of the mesh
        """
        # prepare ofblockmeshdicthelper.BlockMeshDict instance to
        # gather vertices, blocks, faces and boundaries.
        bmd = BlockMeshDict()

        # set metrics
        bmd.set_metric('m')

        # base vertices which are rotated +- 2.5 degrees
        basevs = [
            # Top part
            # Front external main points
            Vertex(self.ex0, self.posYmax, self.ez0, 'v0f'),
            Vertex(self.ex1, self.posYmax, self.ez1, 'v1f'),
            Vertex(self.ex2, self.posYmax, self.ez2, 'v2f'),
            Vertex(self.ex3, self.posYmax, self.ez3, 'v3f'),
            Vertex(self.ex4, self.posYmax, self.ez4, 'v4f'),
            Vertex(self.ex5, self.posYmax, self.ez5, 'v5f'),
            Vertex(self.ex6, self.posYmax, self.ez6, 'v6f'),
            # Back external main points
            Vertex(self.ex0, self.negYmax, self.ez0, 'v0b'),
            Vertex(self.ex1, self.negYmax, self.ez1, 'v1b'),
            Vertex(self.ex2, self.negYmax, self.ez2, 'v2b'),
            Vertex(self.ex3, self.negYmax, self.ez3, 'v3b'),
            Vertex(self.ex4, self.negYmax, self.ez4, 'v4b'),
            Vertex(self.ex5, self.negYmax, self.ez5, 'v5b'),
            Vertex(self.ex6, self.negYmax, self.ez6, 'v6b'),
            # Capsule vertices self.posYmax
            Vertex(self.cx1, self.posYmax, self.cz1, 'c1f' ),
            Vertex(self.cx12,self.posYmax, self.cz12,'c12f'),
            Vertex(self.cx2, self.posYmax, self.cz2, 'c2f' ),
            Vertex(self.cx3, self.posYmax, self.cz3, 'c3f' ),
            Vertex(self.cx4, self.posYmax, self.cz4, 'c4f' ),
            Vertex(self.cx5, self.posYmax, self.cz5, 'c5f' ),
            Vertex(self.cx6, self.posYmax, self.cz6, 'c6f' ),
            # Capsule vertices self.negYmax
            Vertex(self.cx1, self.negYmax, self.cz1, 'c1b' ),
            Vertex(self.cx12,self.negYmax, self.cz12,'c12b'),
            Vertex(self.cx2, self.negYmax, self.cz2, 'c2b' ),
            Vertex(self.cx3, self.negYmax, self.cz3, 'c3b' ),
            Vertex(self.cx4, self.negYmax, self.cz4, 'c4b' ),
            Vertex(self.cx5, self.negYmax, self.cz5, 'c5b' ),
            Vertex(self.cx6, self.negYmax, self.cz6, 'c6b' ),

            # Bottom part
            # Front external main points
            Vertex(self.ex7,  self.posYmax, self.ez7,   'v7f' ),
            Vertex(self.ex8,  self.posYmax, self.ez8,   'v8f' ),
            Vertex(self.ex9,  self.posYmax, self.ez9,   'v9f' ),
            Vertex(self.ex10, self.posYmax, self.ez10,  'v10f'),
            # Back external main points
            Vertex(self.ex7,  self.negYmax, self.ez7,   'v7b' ),
            Vertex(self.ex8,  self.negYmax, self.ez8,   'v8b' ),
            Vertex(self.ex9,  self.negYmax, self.ez9,   'v9b' ),
            Vertex(self.ex10, self.negYmax, self.ez10,  'v10b'),
            # Capsule vertices self.posXmax
            Vertex(self.cx7, self.posYmax, self.cz7, 'c7f' ),
            Vertex(self.cx8, self.posYmax, self.cz8, 'c8f' ),
            Vertex(self.cx9, self.posYmax, self.cz9, 'c9f' ),
            Vertex(self.cx10,self.posYmax, self.cz10,'c10f'),
            # Capsule vertices self.negYmax
            Vertex(self.cx7, self.negYmax, self.cz7, 'c7b' ),
            Vertex(self.cx8, self.negYmax, self.cz8, 'c8b' ),
            Vertex(self.cx9, self.negYmax, self.cz9, 'c9b' ),
            Vertex(self.cx10,self.negYmax, self.cz10,'c10b'),
        ]

        # adds vertices to the class
        for v in basevs:
            bmd.add_vertex(
                round(v.x, 4),
                round(v.y, 4),
                round(v.z, 4),
                v.name
            )
        b1 = bmd.add_hexblock(
            ('v2f', 'c2f', 'c2b', 'v2b', 'v1f', 'c1f', 'c1b', 'v1b'),
            (self.Nx, self.Ny, int(0.5*self.Nz * 87.3/200.5)),
            'b1',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )
        b2 = bmd.add_hexblock(
            ('v3f', 'c3f', 'c3b', 'v3b', 'v2f', 'c2f', 'c2b', 'v2b'),
            (self.Nx, self.Ny, int(0.5*self.Nz * 56.6/200.5)+2),
            'b2',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )

        b3 = bmd.add_hexblock(
            ('v4f', 'c4f', 'c4b', 'v4b', 'v3f', 'c3f', 'c3b', 'v3b'),
            (self.Nx, self.Ny, int(0.5*self.Nz * 56.6/200.5)-1),
            'b3',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )
        b4 = bmd.add_hexblock(
            ('v5f', 'c5f', 'c5b', 'v5b', 'v4f', 'c4f', 'c4b', 'v4b'),
            (self.Nx, self.Ny, int((0.5*self.Nz * 113.1/233))+4),
            'b4',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )

        b5 = bmd.add_hexblock(
            ('v6f', 'c6f', 'c6b', 'v6b', 'v5f', 'c5f', 'c5b', 'v5b'),
            (self.Nx, self.Ny, int(0.5*self.Nz * 120/233)-4),
            'b5',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )

        b6 = bmd.add_hexblock(
            ('v7f', 'c7f', 'c7b', 'v7b', 'v6f', 'c6f', 'c6b', 'v6b'),
            (self.Nx, self.Ny, int(0.5*self.Nz * 120/233)-4),
            'b6',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )

        b7 = bmd.add_hexblock(
            ('v8f', 'c8f', 'c8b', 'v8b', 'v7f', 'c7f', 'c7b', 'v7b'),
            (self.Nx, self.Ny, int(0.5*self.Nz * 113.1/233)+4),
            'b7',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )
        b8 = bmd.add_hexblock(
            ('v9f', 'c9f', 'c9b', 'v9b', 'v8f', 'c8f', 'c8b', 'v8b'),
            (self.Nx, self.Ny, int(0.5*self.Nz * 56.6/200.5)-1),
            'b8',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )

        b9 = bmd.add_hexblock(
            ('v10f', 'c10f', 'c10b', 'v10b', 'v9f', 'c9f', 'c9b', 'v9b'),
            (self.Nx, self.Ny, int(0.5*self.Nz * 56.6/200.5)+2),
            'b9',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )

        b10 = bmd.add_hexblock(
            ('v1f', 'c1f', 'c1b', 'v1b', 'v10f', 'c10f', 'c10b', 'v10b'),
            (self.Nx, self.Ny, int(0.5*self.Nz * 87.3/200.5)),
            'b10',
            grading=SimpleGrading(self.expandX, self.expandY, self.expandZ)
        )

        ######################################################################
        # Add cirlces shapes
        ######################################################################
        # Capsule leading edge, positive z part
        p12cf = Vertex(
            self.cx12,
            self.posYmax,
            self.cz12,
            'c12f')
        bmd.add_arcedge(('c1f', 'c2f'), 'capsuleFront12', p12cf)

        p12cb = Vertex(
            self.cx12,
            self.negYmax,
            self.cz12,
            'c12b')
        bmd.add_arcedge(('c1b', 'c2b'), 'capsuleBac12', p12cb)

        # Inlet, leading edge, positive z part
        p12ef = Vertex(
            self.ex12,
            self.posYmax,
            self.ez12,
            'e12f')
        bmd.add_arcedge(('v1f', 'v2f'), 'inletFront12', p12ef)

        p12eb = Vertex(
            self.ex12,
            self.negYmax,
            self.ez12,
            'e12b')
        bmd.add_arcedge(('v1b', 'v2b'), 'inletBack12', p12eb)

        # Outlet,
        p45ef = Vertex(
            self.ex45,
            self.posYmax,
            self.ez45,
            'e45f')
        bmd.add_arcedge(('v4f', 'v5f'), 'outletFront45', p45ef)

        p45eb = Vertex(
            self.ex45,
            self.negYmax,
            self.ez45,
            'e45b')
        bmd.add_arcedge(('v4b', 'v5b'), 'outletBack45', p45eb)

        # Outlet bottom, Top part
        p56ef = Vertex(
            self.ex56,
            self.posYmax,
            self.ez56,
            'e56f')
        bmd.add_arcedge(('v5f', 'v6f'), 'outletFront56', p56ef)

        p56eb = Vertex(
            self.ex56,
            self.negYmax,
            self.ez56,
            'e56b')
        bmd.add_arcedge(('v5b', 'v6b'), 'outletBack56', p56eb)

        # Outlet bottom, Bottom part
        p67ef = Vertex(
            self.ex67,
            self.posYmax,
            self.ez67,
            'e67f')
        bmd.add_arcedge(('v6f', 'v7f'), 'outletFront67', p67ef)

        p67eb = Vertex(
            self.ex67,
            self.negYmax,
            self.ez67,
            'e67b')
        bmd.add_arcedge(('v6b', 'v7b'), 'outletBack67', p67eb)

        # Outlet bottom, Bottom part
        p78ef = Vertex(
            self.ex78,
            self.posYmax,
            self.ez78,
            'e78f')
        bmd.add_arcedge(('v7f', 'v8f'), 'outletFront78', p78ef)

        p78eb = Vertex(
            self.ex78,
            self.negYmax,
            self.ez78,
            'e78f')
        bmd.add_arcedge(('v7b', 'v8b'), 'outletBack78', p78eb)

        # Inlet, Top part
        p1011ef = Vertex(
            self.ex1011,
            self.posYmax,
            self.ez1011,
            'e1011f')
        bmd.add_arcedge(('v10f', 'v1f'), 'inletFront1011', p1011ef)

        p1011eb = Vertex(
            self.ex1011,
            self.negYmax,
            self.ez1011,
            'e1011b')
        bmd.add_arcedge(('v10b', 'v1b'), 'inletBack1011', p1011eb)

        # Capsule leading edge, Top part
        p1011cf = Vertex(
            self.cx1011,
            self.posYmax,
            self.cz1011,
            'c1011f')
        bmd.add_arcedge(('c10f', 'c1f'), 'capsuleFront1011', p1011cf)

        p1011cb = Vertex(
            self.cx1011,
            self.negYmax,
            self.cz1011,
            'c1011b')
        bmd.add_arcedge(('c10b', 'c1b'), 'capsuleBack1011', p1011cb)

        ######################################################################
        # Add circles splines
        ######################################################################
        inletSplineF = []
        inletSplineB = []
        for x, z, i in zip(self.ex23, self.ez23, range(len(self.ex23))):
            inletSplineF.append(
                # Inlet front spline: ifs
                Vertex(x, self.posYmax, z, 'ifs' + str(i))
            )
            inletSplineB.append(
                # Inlet back spline: ibs
                Vertex(x, self.negYmax, z, 'ibs' + str(i))
            )
        bmd.add_splineedge(
            ('v2f', 'v3f'),
            'inletFrontSpline',
            inletSplineF)
        bmd.add_splineedge(
            ('v2b', 'v3b'),
            'inletBackSpline',
            inletSplineB)

        outletSplineF = []
        outletSplineB = []
        for x, z, i in zip(self.ex34, self.ez34, range(len(self.ex34))):
            outletSplineF.append(
                # Outlet front spline: ofs
                Vertex(x, self.posYmax, z, 'ofs'+str(i))
            )
            outletSplineB.append(
                # Outlet back spline: obs
                Vertex(x, self.negYmax, z, 'obs'+str(i))
            )
        bmd.add_splineedge(
            ('v4f', 'v3f'),
            'outletFrontSpline',
            outletSplineF)
        bmd.add_splineedge(
            ('v4b', 'v3b'),
            'outletBackSpline',
            outletSplineB)

        outletSplineFbottom = []
        outletSplineBbottom = []
        for x, z, i in zip(self.ex89, self.ez89, range(len(self.ex89))):
            outletSplineFbottom.append(
                Vertex(x, self.posYmax, z, 'ofs'+str(i)))
            outletSplineBbottom.append(
                Vertex(x, self.negYmax, z, 'obs'+str(i)))

        bmd.add_splineedge(
            ('v8f', 'v9f'),
            'outletFrontSplineBottom',
            outletSplineFbottom)
        bmd.add_splineedge(
            ('v8b', 'v9b'),
            'outletBackSplineBottom',
            outletSplineBbottom)

        inletSplineFb = []
        inletSplineBb = []
        for x, z, i in zip(self.ex910, self.ez910, range(len(self.ex910))):
            inletSplineFb.append(
                Vertex(x, self.posYmax, z, 'ifs'+str(i)))
            inletSplineBb.append(
                Vertex(x, self.negYmax, z, 'ibs'+str(i)))

        bmd.add_splineedge(
            ('v10f', 'v9f'),
            'inletFrontSplineb',
            inletSplineFb)
        bmd.add_splineedge(
            ('v10b', 'v9b'),
            'inletBackSplineb',
            inletSplineBb)

        # face element of block can be generated by Block.face method
        # bmd.add_boundary(type, name, faces=[])
        # s,n,e,b,t,w
        # s: south
        # n: north
        # e: east
        # w: west 
        # b: bottom
        # t: top
        bmd.add_boundary('patch', 'inlet', [b1.face('w'),
                                            b2.face('w'),
                                            b3.face('w'),
                                            b8.face('w'),
                                            b9.face('w'),
                                            b10.face('w')])
        bmd.add_boundary('patch', 'outlet', [b4.face('w'),
                                             b5.face('w'),
                                             b6.face('w'),
                                             b7.face('w')])
        bmd.add_boundary('wall', 'wall',  [b1.face('e'),
                                            b2.face('e'),
                                            b3.face('e'),
                                            b4.face('e'),
                                            b5.face('e'),
                                            b6.face('e'),
                                            b7.face('e'),
                                            b8.face('e'),
                                            b9.face('e'),
                                            b10.face('e')])

        # prepare for output
        bmd.assign_vertexid()

        # output
        path = self.filePath
        filename = 'blockMeshDict'
        fullpath = os.path.join(path, filename)

        WriteTxtFile = open(fullpath, "w")
        WriteTxtFile.write(bmd.format())
        WriteTxtFile.close()


if __name__ == "__main__":
    # Capsule radius in meters
    diameter = 0.4
    mesh = Capsule2D(diameter)
    mesh.main_points()
    mesh.inlet_spline()
    mesh.outlet_spline()
    mesh.capsule_points()
    mesh.bottom_points()
    # mesh.to_blockMesh_dict()
    mesh.to_blockMesh_dict2()
    # mesh.plot()
