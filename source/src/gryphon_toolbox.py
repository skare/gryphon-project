# Copyright (C) 2012-2015 - Knut Erik Skare
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

import sys as s
import time as time
from datetime import timedelta
from os import getcwd, makedirs

import dolfin as d
import matplotlib.pyplot as plt
import numpy as np
import ufl as UFL


class gryphon_toolbox(object):
    # -----------------------------------------------------------------
    #
    # 						CONSTRUCTOR
    #
    # -----------------------------------------------------------------

    def __init__(self, T, u, f, bcs=[], tdf=[], tdfBC=[]):
        self.u = u  # Initial condition
        self.bcs = bcs  # Boundary conditions
        self.f = f  # Right hand side
        self.tdf = tdf  # Time dependent functions
        self.tdfBC = tdfBC  # Time dependent functions on the boundary
        self.T = T  # Time domain
        self.t = T[0]  # Current time
        self.tstart = T[0]  # Start of domain
        self.tend = T[len(T) - 1]  # End of domain
        self.dt = (self.tend - self.tstart) / 1000.0  # Initial time step
        self.Q = d.TestFunction(self.u.function_space())  # Trial Function to right hand side
        self.U = d.TrialFunction(self.u.function_space())  # Test Function to right hand side
        self.DT = d.Constant(self.dt)  # Current step size, used in forms
        self.savepath = ''  # Path to store program data if requested
        self.dtmax = (T[len(T) - 1] - T[0]) / 10.0  # Max time step
        self.dtmin = 1e-14  # Min time step
        self.nAcc = 0  # Number of accepted steps
        self.nRej = 0  # Number of rejected steps
        self.accepted_steps = []  # Accepted steps
        self.rejected_steps = [[], []]  # Rejected steps
        self.Feval = 0  # Number of function assemblies
        self.Jeval = 0  # Number of Jacobian assemblies

        # Variables for storing previously accepted / rejected time steps
        # with corresponding estimated local error. This is used in the 
        # Gustafsson stepsize selector.
        self.le_acc = 0  # Rejected local error
        self.le_rej = 0  # Rejected time step
        self.dt_acc = 0  # Accepted local error
        self.dt_rej = 0  # Accepted time step

        self.pfactor = 0.8
        self.stepRejected = False
        self.consecutive_rejects = 0
        self.breakTimeLoop = False

        self.verifyInput()
        self.setupSolver()
        self.setupParameters()

    def setupParameters(self):
        # Create parameter object
        self.parameters = d.Parameters("gryphon")
        self.parameters.add("verbose", False)
        self.parameters.add("drawplot", False)
        if (self.u.split() == ()):
            self.plotcomponents = []
        else:
            self.plotcomponents = range(len(self.u.split()))

        # Parameter set "output" nested under "gryphon"
        self.parameters.add(d.Parameters("output"))
        self.parameters["output"].add("plot", False)
        self.parameters["output"].add("path", "")
        self.parameters["output"].add("imgformat", "png")
        self.parameters["output"].set_range("imgformat", {"png", "jpg", "eps"})
        self.parameters["output"].add("reuseoutputfolder", False)
        self.parameters["output"].add("statistics", False)

        try:
            self.parameters.add(d.Parameters("timestepping"))
            self.parameters["timestepping"].add("dt", self.dt)
            if self.supportsAdaptivity:
                self.parameters["timestepping"].add("adaptive", True)
                self.parameters["timestepping"].add("pessimistic_factor", self.pfactor)
                self.parameters["timestepping"].add("absolute_tolerance", 1e-7)
                self.parameters["timestepping"].add("relative_tolerance", 1e-6)
                self.parameters["timestepping"].add("convergence_criterion", "absolute")
                self.parameters["timestepping"].add("dtmax", self.dtmax)
                self.parameters["timestepping"].add("dtmin", self.dtmin)

                # Set range for parameter-object
                self.parameters["timestepping"].set_range("convergence_criterion", {"relative", "absolute"})
                self.parameters["timestepping"].set_range("pessimistic_factor", 0.0, 1.0)

                # Select default step size selector and populate choices.
                self.parameters["timestepping"].add("stepsizeselector", "gustafsson")
                self.parameters["timestepping"].set_range("stepsizeselector", {"gustafsson", "standard"})

                # Dictionary which contains the step size selectors.
                # Needed since the FEniCS parameters system can not store function handles
                self.stepsizeselector = {"gustafsson": self.dtGustafsson, "standard": self.dtStandard}
        except Exception as E:
            raise (E)

    def setupSolver(self):
        if type(self.f) == type([]):
            self.n = len(self.f)
            if type(d.lhs(self.f[0]).integrals()[0].integrand()) == type(
                    UFL.constantvalue.Zero()):  # Not particularly elegant. Should check for another solution.
                self.rank = 1
            else:
                self.rank = 2
        else:
            self.n = 1
            if type(d.lhs(self.f).integrals()[0].integrand()) == type(
                    UFL.constantvalue.Zero()):  # Not particularly elegant. Should check for another solution.
                self.rank = 1
            else:
                self.rank = 2

        # Set appropriate solver
        if self.rank == 2:
            d.info("Using LU-solver to solve linear systems.")
            self.linear = True
            self.solver = d.LUSolver()
        else:
            d.info("Using Newton-solver to solve nonlinear systems.")
            self.linear = False
            self.solver = d.NewtonSolver()

    def updateParameters(self):
        self.dt = self.parameters["timestepping"]["dt"]
        if (self.dt <= 0.0):
            self.gryphonError("Unable to start time integration", "Start step size is 0.0",
                              "Set \"parameters['timestepping']['dt']\" to something greater than zero.")
        self.DT.assign(self.dt)

        if self.supportsAdaptivity:
            self.dtmax = self.parameters["timestepping"]["dtmax"]
            self.dtmin = self.parameters["timestepping"]["dtmin"]
            self.pfactor = self.parameters["timestepping"]["pessimistic_factor"]

            if self.parameters["timestepping"]["convergence_criterion"] == "relative":
                self.tol = self.parameters["timestepping"]["relative_tolerance"]
            elif self.parameters["timestepping"]["convergence_criterion"] == "absolute":
                self.tol = self.parameters["timestepping"]["absolute_tolerance"]

    def savePlot(self):
        try:
            import matplotlib.pyplot
        except ImportError:
            print("Package 'matplotlib' not installed. Unable to save plot of selected step sizes.")
            return

        # Create figure object with scaled axis
        fig = matplotlib.pyplot.figure()
        fig.add_axes([0.1, 0.1, 0.8, 0.7])

        # Plot average step size
        matplotlib.pyplot.plot(np.cumsum(self.accepted_steps),
                               np.mean(self.accepted_steps) * np.ones(len(self.accepted_steps)),
                               '--r',
                               label="Avg. step size")
        # Plot step sizes
        matplotlib.pyplot.plot(np.cumsum(self.accepted_steps),
                               self.accepted_steps,
                               '-o',
                               markersize=3.5,
                               label="Selected step sizes")
        # Plot rejected steps
        matplotlib.pyplot.plot(self.rejected_steps[0],
                               self.rejected_steps[1],
                               's',
                               markersize=5.0,
                               label="Rejected step sizes",
                               markerfacecolor='None')

        # Set axis, labels and legend.
        matplotlib.pyplot.xlabel("Time t")
        matplotlib.pyplot.ylabel("Step size dt")

        maxAcc = max(self.accepted_steps)
        if self.nRej > 0:
            maxRej = max(self.rejected_steps[1])
        else:
            maxRej = 0.0

        matplotlib.pyplot.axis(
            [0, max(np.cumsum(self.accepted_steps)), min(self.accepted_steps) * 0.99, max(maxRej, maxAcc) * 1.01])
        matplotlib.pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                 ncol=2, mode="expand", borderaxespad=0.,
                                 title="Step size selector: %s" % self.parameters["timestepping"]["stepsizeselector"])
        matplotlib.pyplot.savefig("%s/steps.%s" % (self.savepath, self.parameters["output"]["imgformat"]))
        print("Plot of accepted time steps successfully written to '%s'." % (self.savepath))

    def gryphonOutput(self, terminateReason):
        """
        Build a text string to be printed to screen / saved as ASCII-table.
        """

        O = "\n"
        O += "  ********************************************\n"
        if terminateReason == "Success":
            O += "      ODE solver terminated successfully!\n"
        else:
            O += "      ODE solver terminated with an error!\n"
        O += "  ********************************************\n"

        if terminateReason == "StationarySolution" and self.nAcc > 1:
            O += "\n"
            O += "  The local error measured in the current time\n"
            O += "  step is zero to machine precision. The solution\n"
            O += "  may have converged into a stationary state.\n"
        if terminateReason == "StationarySolution" and self.nAcc == 1:
            self.gryphonError("Unable to start time integration.",
                              "The local error measured in the first time step is zero to machine precision.",
                              "Increase the initial step size by altering ESDIRK_object.parameters['timestepping']['dt'].")
        O += "  Method used:\t\t\t%s\n" % self.parameters["method"]
        if terminateReason == "Success":
            O += "  Domain: \t\t\t[%s,%s]\n" % (self.tstart, self.tend)
        else:
            O += "  Successful domain:\t\t[%s,%s] of [%s,%s]\n" % (self.tstart, self.t, self.tstart, self.tend)
        O += "  CPU-time:\t\t\t%g\n" % self.cputime
        O += "  Walltime:\t\t\t%s\n" % timedelta(seconds=round(self.walltime))
        if self.supportsAdaptivity and self.parameters["timestepping"]["adaptive"]:
            if not self.linear:
                O += "  Function evaluations:\t\t%s\n" % self.Feval
                O += "  Jacobian evaluations:\t\t%s\n" % self.Jeval
            O += "  Step size selector:\t\t%s\n" % self.parameters["timestepping"]["stepsizeselector"]
            O += "  Pessimistic factor:\t\t%s\n" % self.parameters["timestepping"]["pessimistic_factor"]
            O += "  Convergence criterion:\t%s\n" % self.parameters["timestepping"]["convergence_criterion"]
            O += "  Absolute tolerance:\t\t%g\n" % self.parameters["timestepping"]["absolute_tolerance"]
            O += "  Relative tolerance:\t\t%g\n" % self.parameters["timestepping"]["relative_tolerance"]
            O += "  Number of steps accepted:\t%s (%g%%)\n" % (
                self.nAcc, round(100.0 * self.nAcc / (self.nAcc + self.nRej), 2))
            O += "  Number of steps rejected:\t%s (%g%%)\n" % (
                self.nRej, round(100.0 * self.nRej / (self.nAcc + self.nRej), 2))
            O += "  Maximum step size selected:\t%g\n" % max(self.accepted_steps)
            O += "  Minimum step size selected:\t%g\n" % min(self.accepted_steps)
            O += "  Mean step size:\t\t%g\n" % np.mean(self.accepted_steps)
            O += "  Variance in step sizes:\t%g\n" % np.var(self.accepted_steps)
        else:
            O += "  Step size used (dt):\t\t%g\n" % self.dt
            O += "  Steps used:\t\t\t%g\n" % self.nAcc

        return O

    def saveStatistics(self, terminateReason):
        pa = round(100.0 * self.nAcc / (self.nAcc + self.nRej), 2)
        pr = round(100.0 * self.nRej / (self.nAcc + self.nRej), 2)

        # Only write LaTeX-table if program terminated successfully.
        if terminateReason == "Success":
            f = open("%s/statistics.tex" % self.savepath, "w")
            f.write("\\begin{table}\n")
            f.write("\\begin{tabular}{l|l}\hline\n")
            f.write("CPU/wall time & %g/%s \\\\ \n" % (self.cputime, timedelta(seconds=round(self.walltime))))
            if self.supportsAdaptivity:
                f.write("No.accepted/rejected steps & %s (%s\\%%)/%s (%s\\%%) \\\\ \n" % (self.nAcc, pa, self.nRej, pr))
                f.write("Convergence criterion & %s \\\\ \n" % self.parameters["timestepping"]["convergence_criterion"])
                f.write("Absolute/relative tolerance & %g/%g \\\\ \n" % (
                    self.parameters["timestepping"]["absolute_tolerance"],
                    self.parameters["timestepping"]["relative_tolerance"]))
                f.write("Pessimistic factor & %g \\\\ \n" % self.pfactor)
                f.write("Step size selector & %s \\\\ \n" % self.parameters["timestepping"]["stepsizeselector"])
                if not self.linear:
                    f.write("Function/Jacobian calls & %s/%s \\\\ \n" % (self.Feval, self.Jeval))
                f.write("$t_{min} / t_{max}$ & %g/%g \\\\ \n" % (min(self.accepted_steps), max(self.accepted_steps)))
                f.write("$t_{mean} / t_{var}$ & %g/%g \\\\ \hline \n" % (
                    np.mean(self.accepted_steps), np.var(self.accepted_steps)))
            else:
                f.write(("Step size used & %s \\\\ \n") % self.dt)
                f.write(("No. steps & %s \\\\ \n") % self.nAcc)

            f.write("\end{tabular} \n")
            f.write("\caption{Results using %s on domain [%s,%s].} \n" % (
                self.parameters["method"], self.tstart, self.tend))
            f.write("\end{table} \n")
            f.close()

        # Ascii-table is written regardless of terminate reason.
        f = open("%s/statistics.txt" % self.savepath, "w")
        f.write(self.gryphonOutput(terminateReason))
        f.close()

        print("Job statistics file successfully written to '%s'." % (self.savepath))

    def generateOutput(self, terminateReason):
        if terminateReason != "success":
            print(self.gryphonOutput(terminateReason))
        elif self.parameters["verbose"]:
            print(self.gryphonOutput(terminateReason))
        if self.parameters["output"]["statistics"]:
            self.saveStatistics(terminateReason)
            try:
                if self.supportsAdaptivity and self.parameters["timestepping"]["adaptive"]:
                    self.savePlot()
            except:
                print("** ERROR: Unable to save plot of selected step sizes.\n"
                      "          If you are running in an environment without graphics (i.e. on a computational server), be sure that matplotlib is using the 'Agg'-backend.\n"
                      "          You can set this by writing matplotlib.use('Agg')")

    def dtGustafsson(self, le, p, stepAccepted):
        """
        Input:
          le            - Estimate for local error in current time step
          p             - Order of the error estimate
          stepAccepted  - True if step should be accepted. Otherwise False.
    
        This is the step size selector developed by Karl Gustafsson for
        implicit Runge-Kutta methods.
        """
        k1 = 1.0 / p
        k2 = 1.0 / p

        if stepAccepted:
            if self.nAcc + self.nRej == 1 or self.dt_restricted or self.stepRejected:
                dt = np.power(self.tol / le, 1.0 / p) * self.dt
                self.stepRejected = False
            else:
                dt = (self.dt / self.dt_acc) * np.power(self.tol / le, k2) * np.power(self.le_acc / le, k1) * self.dt
            self.dt_acc = self.dt
            self.le_acc = le
        else:
            if self.consecutive_rejects > 1:
                k_est = np.log(le / self.le_rej) / np.log(self.dt / self.dt_rej)
                if k_est > p:
                    k_est = p
                elif k_est < 0.1:
                    k_est = 0.1
                dt = np.power(self.tol / le, 1.0 / k_est) * self.dt
            else:
                dt = np.power(self.tol / le, 1.0 / p) * self.dt
            self.dt_rej = self.dt
            self.le_rej = le

        self.dt = self.pfactor * dt

    def dtStandard(self, le, p, stepAccepted):
        """
        Input:
          le            - Estimate for local error in current time step
          p             - Order of the error estimate
          stepAccepted  - True if step should be accepted. Otherwise False.
      
        This is the standard step size selector derived from
        asymptotic theory.
        """
        self.dt_acc = self.dt
        self.dt = self.pfactor * np.power(self.tol / le, 1.0 / p) * self.dt

    def verifyStepsize(self):
        """
        Safety net for step size selectors.
        It operates according to the following rules:
    
        Step size can not be:
          - greater than C1*(previous time step)
          - greater than self.dtmax
          - less than C2*(previous time step).
          - less than self.dtmin
          - greater than remaining time domain
        """
        if self.supportsAdaptivity:
            C1 = 1.5
            C2 = 0.1
            if self.parameters['timestepping']['adaptive']:
                if self.dt > self.dtmax:
                    self.dt = self.dtmax
                    self.dt_restricted = True
                elif self.dt_acc > 0.0 and self.dt > C1 * self.dt_acc:
                    self.dt = C1 * self.dt_acc
                    self.dt_restricted = True
                elif self.dt_acc > 0.0 and self.dt < C2 * self.dt_acc:
                    self.dt = C2 * self.dt_acc
                    self.dt_restricted = True
                elif self.dt < self.dtmin:
                    self.gryphonError("Unable to integrate system to user given tolerance tol = %g." % self.tol,
                                      "Minimum step size dt = %g selected." % self.dtmin)
                else:
                    self.dt_restricted = False

        if self.t + self.dt >= self.tend:
            self.dt = self.tend - self.t
            self.breakTimeLoop = True

        self.DT.assign(self.dt)

    def acceptStep(self, le, u):
        """
        Input:
          le - Estimate for local error in current time step.
          u  - Solution in current time step.
      
        This function returns True if the current time step should be
        accepted. Otherwise it returns False.
        """
        if self.parameters['timestepping']['convergence_criterion'] == "relative":
            if le <= max(self.parameters['timestepping']['relative_tolerance'] * norm(u.vector(), 'l2'),
                         self.parameters['timestepping']['absolute_tolerance']):

                # Update tolerance which to select the new stepsize from
                self.tol = max(self.parameters['timestepping']['relative_tolerance'] * norm(u.vector(), 'l2'),
                               self.parameters['timestepping']['absolute_tolerance'])
                return True
            else:
                return False
        elif self.parameters['timestepping']['convergence_criterion'] == "absolute":
            if le <= self.parameters['timestepping']['absolute_tolerance']:
                return True
            else:
                return False

    def figureHandling(self, Init=False, Update=False):
        """
        Input:
          Init    - Initialize plots/savefiles.
          Update  - Updates the initialized plots/savefiles.
        """
        if Init:
            if self.parameters['output']['plot']:
                if self.plotcomponents == []:
                    self.file = d.File('%s/plot/u_0.pvd' % self.savepath)
                    self.file << (self.u, float(self.t))
                else:
                    self.file = [d.File('%s/plot/u%s_0.pvd' % (self.savepath, i)) for i in range(self.n)]
                    for i in self.plotcomponents:
                        self.file[i] << (self.u.split()[i], float(self.t))
            if self.parameters['drawplot']:
                self.plots = [plt.figure() for i in range(self.n)]

        if Update:
            if self.parameters['output']['plot']:
                if self.plotcomponents == []:
                    self.file << (self.u, float(self.t))
                else:
                    for i in self.plotcomponents:
                        self.file[i] << (self.u.split()[i], float(self.t))
            if self.parameters['drawplot']:
                if self.plotcomponents == []:
                    plt.figure(1)
                    plt.cla()
                    d.plot(self.u)
                    plt.ion()
                    plt.pause(0.001)
                else:
                    for figureCounter in range(0, self.n):
                        plt.figure(figureCounter + 1) # Must start with figure 1, not figure 0
                        plt.cla()
                        d.plot(self.u.split()[figureCounter])
                        plt.ion()
                        plt.pause(0.001)

    def solve(self):
        if self.parameters["output"]["statistics"] or self.parameters["output"]["plot"]:
            self.verifyOutputPath()
        self.cpustart = time.clock()
        self.wallclockstart = time.time()

        # Used when estimating program run time
        self.timestepTimer = 0.0
        self.updateParameters()
        if self.supportsAdaptivity:
            self.currentStepsizeSelector = self.stepsizeselector[self.parameters['timestepping']['stepsizeselector']]

    def terminateTimeLoop(self, terminateReason):
        self.cputime = time.clock() - self.cpustart
        self.walltime = time.time() - self.wallclockstart
        self.generateOutput(terminateReason)

    def gryphonError(self, task, reason, remedy=False):
        EM = "\n----------------------------------------------------------------------------"
        EM += "\n                *** Gryphon encountered an error ***"
        EM += "\n----------------------------------------------------------------------------"
        EM += "\n*** " + "Error : " + task
        EM += "\n*** " + "Reason: " + reason
        if remedy:
            EM += "\n*** " + "Remedy: " + remedy
        EM += "\n----------------------------------------------------------------------------"
        EM += "\n"
        raise RuntimeError(EM)

    def verifyOutputPath(self):
        path = self.parameters['output']['path']
        cwd = getcwd()

        if path == "":
            path = s.argv[0].replace(".py", "") + "_gryphon_data"

        try:
            if path[0] == "/":
                # Absolute path
                self.savepath = path
            else:
                # Relative path
                self.savepath = getcwd() + "/" + path

            makedirs(self.savepath)
        except OSError as E:
            if E.errno == 13:
                self.gryphonError(
                    str(E),
                    "User running script has insuficient access to specified path.",
                    remedy="Change output folder to a folder you know you own.")
            elif E.errno == 17:
                # Folder already exists, ask user if it is OK to reuse
                if self.parameters["output"]["reuseoutputfolder"]:
                    # It is OK to reuse the savepath. Just pass.
                    pass
                else:
                    userResponse = False
                    while not userResponse:
                        print(
                            "!! WARNING: The folder '" + self.savepath + "' already exists.\nPress Y and ENTER to reuse folder (FILES MIGHT BE OVERWRITTEN!!). Press N and ENTER to exit.")
                        print(
                            "To always reuse the folder, set the option 'parameters['output']['reuseoutputfolder'] = True'")
                        userinput = input('Y/N: ')
                        if userinput.upper() == 'N':
                            s.exit(0)
                        elif userinput.upper() == 'Y':
                            userResponse = True
            else:
                raise (E)

        # Do a final test to check whether a file can be created in the
        # desired folder.
        try:
            finaltest = open(self.savepath + "/" + ".gryphonwritetest", "w")
            finaltest.close()
        except IOError as E:
            if E.errno == 13:
                self.gryphonError(
                    "Unable to write to folder '" + self.savepath + "'",
                    "User running script has insuficient access to specified path.",
                    remedy="Change output folder to a folder you know you own.")
            else:
                raise (E)

    def printProgress(self, timeEstimate=False, rejectedStep=False):
        n = 20  # Width of progress bar
        totalTime = self.tend - self.tstart
        elapsedTime = self.t - self.tstart
        progress = round(100.0 * elapsedTime / totalTime, 1)
        dots = int(round(n * progress / 100.0))

        progressBar = "|" + "=" * dots + ">" + "." * (n - dots) + "| " + (str(progress) + " %").ljust(7) + (
                " t=%g" % (self.t)).ljust(15)
        if rejectedStep:
            progressBar += "Step rejected, dt=%g" % self.dt

        if timeEstimate and (self.tstart + self.t) / (self.tend - self.tstart) > 0.01 and not rejectedStep:
            progressBar += "Completion in ~ %s" % (timeEstimate())

        print(progressBar)

    def verifyInput(self):
        if type(self.T) != list:
            self.gryphonError("Error in first required argument: Time domain.",
                              "Domain must be speficied as a list with two floats representing t_start and t_end.")
        elif self.T[1] - self.T[0] < 0:
            self.gryphonError("Error in first required argument: Time domain.",
                              "Specified time domain [%s,%s] is negative." % (self.T[0], self.T[1]))

        if type(self.u) != d.Function:
            self.gryphonError("Error in second required argument: Initial condition.",
                              "The initial condition must be given as a Function-object.")

        if type(self.f) != list:
            self.f = [self.f]
        if any([type(self.f[i]) != UFL.form.Form for i in range(len(self.f))]):
            self.gryphonError("Error in third required argument: Right hand side.",
                              "The right hand side must be given as either a single Form-object or a list of Form-objects.")

        if self.bcs != []:
            if type(self.bcs) != list:
                self.bcs = [self.bcs]
            if any([type(self.bcs[i]) != (d.DirichletBC) for i in range(len(self.bcs))]):
                self.gryphonError("Error in keyword argument: 'bcs'.",
                                  "Boundary conditions must be given as either a single Dirichlet object" +
                                  "\n            or a list of Dirichlet objects.")

        if self.tdf != []:
            if type(self.tdf) != list:
                self.tdf = [self.tdf]

        if self.tdfBC != []:
            if type(self.tdfBC) != list:
                self.tdf = [self.tdfBC]


class nonlinearStage(d.NonlinearProblem):
    def __init__(self, a, l, bcs):
        d.NonlinearProblem.__init__(self)
        self.a = a
        self.l = l
        self.bcs = bcs
        self.Jeval = 0
        self.Feval = 0

    def F(self, b, x):
        self.Feval += 1
        d.assemble(self.l, tensor=b)
        for i in self.bcs:
            i.apply(b, x)

    def J(self, A, x):
        self.Jeval += 1
        d.assemble(self.a, tensor=A)
        for i in self.bcs:
            i.apply(A)


class linearStage:
    def __init__(self, a, l, bcs, solver):
        self.a = a
        self.l = l
        self.bcs = bcs
        self.solver = solver

    def solve(self, X):
        A = d.assemble(self.a)
        b = d.assemble(self.l)
        for i in self.bcs:
            i.apply(A)
            i.apply(b)
        self.solver.solve(A, X, b)
