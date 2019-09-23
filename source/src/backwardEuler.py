# Copyright (C) 2012-2019 - Knut Erik Skare
#
# This file is part of Gryphon.
#
# Gryphon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gryphon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gryphon. If not, see <http://www.gnu.org/licenses/>.

import time as time
from datetime import timedelta
from functools import reduce

import dolfin as d
import ufl as ufl

from .gryphon_toolbox import gryphon_toolbox, linearStage, nonlinearStage


class backwardEuler(gryphon_toolbox):
    def __init__(self, T, u, f, bcs=[], tdf=[], tdfBC=[]):
        # The current backward Euler implementation does not provide
        # an estimate for the local error and does thus not support
        # adaptive step size selection.
        self.supportsAdaptivity = False

        # Call toolbox constructor
        gryphon_toolbox.__init__(self, T, u, f, bcs, tdf, tdfBC)

        self.parameters.add("method", "Backward Euler")
        self.parameters.set_range("method", {"Backward Euler"})

    def getLinearVariationalForms(self, X):
        # Generate copies of time dependent functions
        self.tdfButcher = [[] for i in range(len(self.tdf))]
        for j in range(0, len(self.tdf)):
            if self.tdf[j].__class__.__name__ == "CompiledExpression":
                self.tdfButcher[j].append(d.Expression(self.tdf[j].cppcode, t=self.tstart))
            else:
                self.tdfButcher[j].append(self.tdf[j])

        if self.n == 1:
            # Add differential equation
            L = [self.U * self.Q * d.dx - self.u * self.Q * d.dx]
            R = {}
            for k in range(0, len(self.tdf)):
                R[self.tdf[k]] = self.tdfButcher[k][0]

            L[0] -= self.DT * ufl.replace(self.f[0], R)

        else:
            # Add differential equations
            L = [reduce((lambda x, y: x + y),
                        [self.U[alpha] * self.Q[alpha] * d.dx - self.u[alpha] * self.Q[alpha] * d.dx for alpha in
                         range(self.n)])]
            for alpha in range(0, self.n):
                R = {}
                for k in range(0, len(self.tdf)):
                    R[self.tdf[k]] = self.tdfButcher[k][0]

                L[0] -= self.DT * ufl.replace(self.f[alpha], R)
        return L

    def getNonlinearVariationalForms(self, X):
        # Generate copies of time dependent functions
        self.tdfButcher = [[] for i in range(len(self.tdf))]
        for j in range(0, len(self.tdf)):
            if self.tdf[j].__class__.__name__ == "CompiledExpression":
                self.tdfButcher[j].append(d.Expression(self.tdf[j].cppcode, t=self.tstart))
            else:
                self.tdfButcher[j].append(self.tdf[j])

        if self.n == 1:
            # Add differential equations
            L = [X[0] * self.Q * d.dx - self.u * self.Q * d.dx]
            replaceDict = {self.u: X[0]}
            for k in range(0, len(self.tdf)):
                replaceDict[self.tdf[k]] = self.tdfButcher[k][0]
            L[0] -= self.DT * replace(self.f[0], replaceDict)
        else:
            # Add differential equations
            L = [reduce((lambda x, y: x + y),
                        [X[0][alpha] * self.Q[alpha] * d.dx - self.u[alpha] * self.Q[alpha] * d.dx for alpha in
                         range(self.n)])]
            for alpha in range(self.n):
                replaceDict = {self.u: X[0]}
                for k in range(len(self.tdf)):
                    replaceDict[self.tdf[k]] = self.tdfButcher[k][j]
                L[0] -= self.DT * ufl.replace(self.f[alpha], replaceDict)
        return L

    def solve(self):
        super(backwardEuler, self).solve()

        # Array for storing the stage values
        X = [self.u.copy(deepcopy=True)]

        # Get the variational linear/nonlinear variational forms
        # and embed them in respective solver class
        if self.linear:
            l = self.getLinearVariationalForms(X)
            p = [linearStage(d.lhs(l[0]), d.rhs(l[0]), self.bcs, self.solver)]
        else:
            l = self.getNonlinearVariationalForms(X)
            a = [d.derivative(l[0], X[0], self.U)]
            p = [nonlinearStage(a[0], l[0], self.bcs)]

        # Initialize save/plot of function(s)
        self.figureHandling(Init=True)

        # Time stepping loop
        while True:
            timestepStart = time.time()

            # Update time dependent functions
            for i in range(len(self.tdfButcher)):
                for j in range(0, 1):
                    self.tdfButcher[i][j].t = self.t + self.dt

            # Update time dependent functions on boundary
            for F in self.tdfBC:
                F.t = self.t + self.dt

            # Solve for implicit stages
            if self.linear:
                p[0].solve(X[0].vector())
            else:
                self.solver.solve(p[0], X[0].vector())

            # Constant step size integration
            self.u.vector()[:] = X[0].vector()[:]
            self.t += self.dt
            self.nAcc += 1
            if self.parameters['verbose']:
                self.printProgress(self.estimateRuntime)

            # Update / save plots
            self.figureHandling(Update=True)

            # Break if this is final time step
            if self.breakTimeLoop:
                terminateReason = "Success"
                break

            self.timestepTimer = time.time() - timestepStart
            self.verifyStepsize()

        super(backwardEuler, self).terminateTimeLoop(terminateReason)

    def estimateRuntime(self):
        steps = (self.tend - self.t) / self.dt
        return str(timedelta(seconds=round(self.timestepTimer * steps)))
