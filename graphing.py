import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chisqprob


class Tests:
    def __init__(self, data_points=np.array([0, 0])):
        self.data_points = data_points

    def chi_test(self, x):
        m = x[0]
        c = x[1]

        chi_square_sum = 0
        for i in range(len(self.data_points)):  # sums the chi^2 values for each point
            y_calc = m * self.data_points[i, 0] + c  # calculates a y value using data points and a given m and c
            if self.data_points[i, 3] != 0:
                chi_square = ((self.data_points[i, 1] - y_calc) / self.data_points[i, 3])**2
            else:
                chi_square = ((self.data_points[i, 1] - y_calc) / self.data_points[i, 1])**2
            chi_square_sum += chi_square
        return chi_square_sum

    def correlation_coef(self):  # to be tested
        list_1, list_2 = self.data_points[0], self.data_points[1]
        return np.corrcoef(list_1, list_2)

    def chi_probability(self, chi_value):
        dof = len(self.data_points)-1
        return chisqprob(chi_value, dof)


class Optimise:
    def __init__(self, start_params, test):
        self.start_params = start_params
        self.test = test

    def scipy_optimize(self):
        return minimize(fun=self.test, x0=self.start_params)


class Plotting:
    def __init__(self, data_points=np.array([0, 0])):
        self.data_points = data_points

    def load_data(self):
        data = str(input("Data: "))
        self.data_points = np.loadtxt(data)

    def plot_residuals(self, x, show=False, save=False, title='', x_title='', y_title=''):
        y_calc = [self.data_points[i, 0]*x[0] + x[1] for i in range(len(self.data_points))]
        residual = [self.data_points[i, 1] - y_calc[i] for i in range(len(self.data_points))]

        for i in range(len(self.data_points)):
            plt.errorbar(x=self.data_points[i, 0], y=residual[i], yerr=self.data_points[i, 3],
                         fmt='o', capsize=3, ecolor='C0', color='C0', elinewidth=0.5)
        plt.axhline(0, linestyle='--', color=(0, 0, 0), linewidth=1)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.title(title)
        # plt.box(on=None)  # removes outline box
        if show:
            plt.show()
        if save:
            title = "_".join(title.split())
            plt.savefig(str(title) + '.png')
        return residual

    def cross_axis(self, residual):
        count = 0
        for i in range(len(self.data_points)):
            if abs(residual[i]) - self.data_points[i, 3] <= 0:
                count += 1
        return count

    def plot_points(self, trendline=False, show=False, title='', x_title='', y_title='', save=False):
        for i in range(len(self.data_points)):
            plt.errorbar(self.data_points[i, 0], self.data_points[i, 1],
                         xerr=self.data_points[i, 2], yerr=self.data_points[i, 3],
                         fmt='o', capsize=3, ecolor='C0', color='C0', elinewidth=0.5)
            plt.title(title)
            plt.xlabel(x_title)
            plt.ylabel(y_title)

        if trendline.any():
            low_end = self.data_points.min()
            high_end = self.data_points.max()
            x = np.linspace(low_end*1.1, high_end*1.1, 10)
            y = [trendline[0]*x[i] + trendline[1] for i in range(len(x))]
            plt.plot(x, y, '-.', color=(1, 165/255, 0))
            # plt.legend(['$y = %fx+%f$' % (trendline[0], trendline[1])])  # display trend line equation
        if show:
            plt.show()
        if save:
            title = "_".join(title.split())
            title = str(input("Save as: "))
            plt.savefig(str(title)+'.png')

    def print_output(self, chi=False, cross_axis=False, cor_coef=False, chi_confidence=False, mc=False):
        if chi:
            print("Chi^2: %.2f. Degrees of freedom: %d" % (chi, len(self.data_points)-1))
            if chi_confidence:
                print("Giving a confidence value (critical value) of: %.4f" % chi_confidence)
        if mc.any():
            print("Best fit line equation: y = %.3fx + %.3f" % (mc[0], mc[1]))
        if cross_axis:
            percent = 100 * (cross_axis / len(self.data_points))
            print(cross_axis, '/', len(self.data_points), ', (%.2f%%)' % percent, ' data points cross the axis.')
        if cor_coef:
            print("Correlation Coefficient between x and y data: %.2f" % cor_coef)


def main():

    plotting = Plotting()
    plotting.load_data()
    tests = Tests(plotting.data_points)
    start_params = np.array([1, 0])
    optimize = Optimise(start_params, tests.chi_test)
    mc = optimize.scipy_optimize()  # this is the large output
    chi_squared = mc.fun
    title = str(input("Graph title: "))
    x_lable = str(input("x_lable (remember $\chi^2$.. like latex): "))
    y_lable = str(input("y_lable: "))
    plotting.plot_points(trendline=mc.x, show=True, title=title, x_title=x_lable, y_title=y_lable, save=True)  # trendline = mc.x
    residual = plotting.plot_residuals(x=mc.x, show=True, title='Residual Plot of %s' % y_lable, x_title=x_lable, y_title="Residual in %s" % y_lable, save=True)
    cross_axis = plotting.cross_axis(residual)
    cor_coef = tests.correlation_coef()[0, 1]
    chi_confidence = tests.chi_probability(chi_squared)

    plotting.print_output(chi_squared, cross_axis, cor_coef, chi_confidence, mc.x)







main()
