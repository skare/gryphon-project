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

import time
from datetime import timedelta
from functools import reduce

import numpy as np
from .ESDIRK_tables import getTable
from dolfin import dx, action, lhs, rhs, norm, derivative, Expression
from .gryphon_toolbox import gryphon_toolbox, linearStage, nonlinearStage
from ufl import form, replace


class ESDIRK(gryphon_toolbox):
    def __init__(self, T, u, f, g=[], bcs=[], tdf=[], tdfBC=[]):
        # The ESDIRK methods provides an estimate for the local error
        # and thus supports adaptive step size selection.
        self.supportsAdaptivity = True

        # Call toolbox constructor
        gryphon_toolbox.__init__(self, T, u, f, bcs, tdf, tdfBC)

        self.parameters.add("method", "ESDIRK43a")
        self.parameters.set_range("method", {"ESDIRK43a", "ESDIRK43b", "ESDIRK32a", "ESDIRK32b"})
        self.parameters['timestepping'].add("inconsistent_initialdata", False)

        # Do input verification on any algebraic components.
        self.g = g
        if type(self.g) != list:
            self.g = [self.g]
            if any([type(self.g[i]) != form.Form for i in range(len(self.g))]):
                self.gryphonError("Error in keyword argument: 'g'.",
                                  "The right hand side must be given as either a single Form-object or a list of Form-objects.")

        self.m = len(self.g)  # Number of algebraic components
        self.n = self.n + self.m  # Total number of PDEs in system

    def getLinearVariationalForms(self, B, X):
        # Get the number of stage values
        s = B.shape[0]
        # Generate copies of time dependent functions
        self.tdfButcher = [[] for i in range(len(self.tdf))]
        for j in range(0, len(self.tdf)):
            for i in range(0, s):
                if self.tdf[j].__class__.__name__ == "CompiledExpression":
                    self.tdfButcher[j].append(Expression(self.tdf[j].cppcode, t=self.tstart))
                else:
                    self.tdfButcher[j].append(self.tdf[j])

        if self.n == 1:
            # Add differential equation
            L = [self.U * self.Q * dx - self.u * self.Q * dx for j in range(s - 1)]
            for j in range(0, s - 1):
                R = {}
                for k in range(0, len(self.tdf)):
                    R[self.tdf[k]] = self.tdfButcher[k][j + 1]
                L[j] -= self.DT * B[1, 1] * replace(self.f[0], R)

            for i in range(0, s - 1):
                for j in range(0, i + 1):
                    R = {}
                    for k in range(0, len(self.tdf)):
                        R[self.tdf[k]] = self.tdfButcher[k][j]
                    L[i] -= self.DT * B[i + 1, j] * action(replace(self.f[0], R), X[j])
        else:
            # Add differential equations
            L = [reduce((lambda x, y: x + y),
                        [self.U[alpha] * self.Q[alpha] * dx - self.u[alpha] * self.Q[alpha] * dx for alpha in
                         range(self.n - self.m)]) for i in range(s - 1)]
            for alpha in range(0, self.n - self.m):
                for i in range(s - 1):
                    R = {}
                    for k in range(0, len(self.tdf)):
                        R[self.tdf[k]] = self.tdfButcher[k][i + 1]
                    L[i] -= self.DT * B[1, 1] * replace(self.f[alpha], R)
                    for j in range(i + 1):
                        R = {}
                        for k in range(0, len(self.tdf)):
                            R[self.tdf[k]] = self.tdfButcher[k][j]
                        L[i] -= self.DT * B[i + 1, j] * action(replace(self.f[alpha], R), X[j])

            # Add algebraic equations
            for beta in range(self.m):
                for i in range(s - 1):
                    R = {}
                    for k in range(len(self.tdf)):
                        R[self.tdf[k]] = self.tdfButcher[k][i + 1]
                    L[i] += replace(self.g[beta], R)

        return L

    def getNonlinearVariationalForms(self, B, X):
        s = B.shape[0]

        # Generate copies of time dependent functions
        self.tdfButcher = [[] for i in range(len(self.tdf))]
        for j in range(0, len(self.tdf)):
            for i in range(0, s):
                if self.tdf[j].__class__.__name__ == "CompiledExpression":
                    self.tdfButcher[j].append(Expression(self.tdf[j].cppcode, t=self.tstart))
                else:
                    self.tdfButcher[j].append(self.tdf[j])

        if self.n == 1:
            # Add differential equations
            L = [X[j + 1] * self.Q * dx - self.u * self.Q * dx for j in range(s - 1)]
            for i in range(s - 1):
                for j in range(s):
                    replaceDict = {self.u: X[j]}
                    for k in range(0, len(self.tdf)):
                        replaceDict[self.tdf[k]] = self.tdfButcher[k][j]
                    L[i] -= self.DT * B[i + 1, j] * replace(self.f[0], replaceDict)
        else:
            # Add differential equations
            L = [reduce((lambda x, y: x + y),
                        [X[j + 1][alpha] * self.Q[alpha] * dx - self.u[alpha] * self.Q[alpha] * dx for alpha in
                         range(self.n - self.m)]) for j in range(s - 1)]
            for alpha in range(self.n - self.m):
                for i in range(s - 1):
                    for j in range(s):
                        replaceDict = {self.u: X[j]}
                        for k in range(len(self.tdf)):
                            replaceDict[self.tdf[k]] = self.tdfButcher[k][j]
                        L[i] -= self.DT * B[i + 1, j] * replace(self.f[alpha], replaceDict)

            # Add algebraic equations
            for beta in range(self.m):
                for i in range(s - 1):
                    replaceDict = {self.u: X[i + 1]}
                    for k in range(len(self.tdf)):
                        replaceDict[self.tdf[k]] = self.tdfButcher[k][i + 1]
                    L[i] += replace(self.g[beta], {self.u: X[i + 1]})
        return L

    def solve(self):
        super(ESDIRK, self).solve()

        B = getTable(self.parameters['method'])
        s = B['tableau'].shape[0]

        # Array for storing the stage values
        X = [self.u.copy(deepcopy=True) for i in range(s)]

        # Get the variational linear/nonlinear variational forms
        # and embed them in respective solver class
        if self.linear:
            l = self.getLinearVariationalForms(B['tableau'], X)
            p = [linearStage(lhs(l[j]), rhs(l[j]), self.bcs, self.solver) for j in range(s - 1)]
        else:
            l = self.getNonlinearVariationalForms(B['tableau'], X)
            a = [derivative(l[i], X[i + 1], self.U) for i in range(s - 1)]
            p = [nonlinearStage(a[i], l[i], self.bcs) for i in range(s - 1)]

        # Initialize save/plot of function(s)
        self.figureHandling(Init=True)

        # Time stepping loop
        while True:
            timestepStart = time.time()
            # Explicit first stage
            X[0].vector()[:] = self.u.vector()[:]

            if self.parameters['timestepping']['adaptive']:
                S = s - 1
            elif not self.parameters['timestepping']['adaptive']:
                S = B['advSt']

            if self.parameters['timestepping']['inconsistent_initialdata'] and (self.nAcc + self.nRej) == 0:
                initdt = self.dt
                self.dt = self.dtmin
                self.DT.assign(self.dtmin)
            elif self.parameters['timestepping']['inconsistent_initialdata'] and (self.nAcc + self.nRej) == 1:
                self.dt = initdt
                self.DT.assign(initdt)

            # Update time dependent functions
            for i in range(len(self.tdfButcher)):
                for j in range(0, s):
                    self.tdfButcher[i][j].t = self.t + sum(B['tableau'][j, :]) * self.dt

            # Update time dependent functions on boundary
            for j in range(0, S):
                for F in self.tdfBC:
                    F.t = self.t + sum(B['tableau'][j + 1, :]) * self.dt

                # Solve for implicit stages
                if self.linear:
                    p[j].solve(X[j + 1].vector())
                else:
                    self.solver.solve(p[j], X[j + 1].vector())

            # Adaptive step size integration
            if self.parameters['timestepping']['adaptive']:
                le = norm(X[s - 1].vector() - X[s - 2].vector(), 'l2')
                if self.acceptStep(le, X[B['advSt']]):
                    self.u.vector()[:] = X[B['advSt']].vector()[:]
                    if not self.breakTimeLoop:
                        self.accepted_steps.append(self.dt)
                    self.t += self.dt
                    self.nAcc += 1
                    stepAccepted = True
                    self.consecutive_rejects = 0
                    if self.parameters['verbose']:
                        self.printProgress(self.estimateRuntime)
                else:
                    self.nRej += 1
                    self.consecutive_rejects += 1
                    stepAccepted = False
                    self.stepRejected = True
                    self.rejected_steps[0].append(self.t + self.dt)
                    self.rejected_steps[1].append(self.dt)
                    if self.parameters['verbose']:
                        self.printProgress(rejectedStep=True)

            # Constant step size integration
            elif not self.parameters['timestepping']['adaptive']:
                self.u.vector()[:] = X[B['advSt']].vector()[:]
                self.t += self.dt
                self.nAcc += 1
                if self.parameters['verbose']:
                    self.printProgress(self.estimateRuntime)

            # Update / save plots
            self.figureHandling(Update=True)

            # Select new step size
            if self.parameters['timestepping']['adaptive']:
                try:
                    self.currentStepsizeSelector(le, B['order'], stepAccepted)
                except ZeroDivisionError:
                    terminateReason = "StationarySolution"
                    break

            # Break if this is final time step
            if self.breakTimeLoop:
                terminateReason = "Success"
                break

            self.timestepTimer = time.time() - timestepStart
            self.verifyStepsize()

        # Generate output, sum up function evaluations and terminate program
        if not self.linear:
            for i in range(len(p)):
                self.Feval += p[i].Feval
                self.Jeval += p[i].Jeval
        super(ESDIRK, self).terminateTimeLoop(terminateReason)

    def estimateRuntime(self):
        if self.parameters['timestepping']['adaptive']:
            steps = (self.tend - self.t) / np.mean(self.accepted_steps)
        else:
            steps = (self.tend - self.t) / self.dt
        return str(timedelta(seconds=round(self.timestepTimer * steps)))
