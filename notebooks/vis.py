import os
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


NPTS = 100
LABEL_Y_OFFSET_FACTOR = 30.0
FIGS_DIR = "../assets/images/"


class COLORS:
    blue = "#4257B2"
    light_blue = "#A1C4FD"
    cyan = "#3CCFCF"
    green = "#388E34"
    light_green = "#28CC7D"
    dark_green = "#006060"
    yellow = "#FFCD1F"
    salmon = "#FF725B"
    red = "#FB3640"
    dark_red = "#AE2024"
    purple = "#8842C0"
    gray = "#687174"
    dark_gray = "#455357"
    light_gray = "#C0CACE"
    brown = "#665000"


CONTROL_COLOR = COLORS.blue
VARIATION_COLOR = COLORS.green
DIFF_COLOR = COLORS.dark_gray
RESULTS_FIGSIZE = (15, 10)


# Plotting Utils (more-or-less copy/pasted from abracadabra)
class Plottable:
    def __init__(self, label=None, color=None):
        self.label = label
        self.color = color


class Pdf(Plottable):
    """
    Base class for plotting probability density functions.
    """

    def __init__(self, fill=True, *args, **kwargs):
        super(Pdf, self).__init__(*args, **kwargs)
        self.fill = fill

    def density(self, xs):
        """
        Evaluate
        """
        raise NotImplementedError("Implement Me")

    def xgrid(self):
        """
        Return the default x-values for plotting
        """
        raise NotImplementedError("Implement Me")

    def ppf(self, x):
        return self.dist.ppf(x)

    def cdf(self, x):
        return self.dist.cdf(x)

    def get_series(self):
        xs = self.xgrid().flatten()
        ys = self.density(xs)
        return xs, ys

    def plot(self, **plot_args):
        (
            xs,
            ys,
        ) = self.get_series()
        plt.plot(xs, ys, label=self.label, color=self.color, **plot_args)
        if self.fill:
            self.plot_area(xs, ys)

    def plot_area(self, xs=None, ys=None, color=None, alpha=0.25, label=None):
        xs = self.xgrid().flatten() if xs is None else xs
        ys = self.density(xs) if ys is None else ys
        color = self.color if color is None else color
        plt.fill_between(xs, ys, color=color, alpha=alpha, label=label)

    def sample(self, size):
        return self.dist.rvs(size=size)


class KdePdf(Pdf):
    """
    Estimate the shape of a PDF using a kernel density estimate.
    """

    def __init__(self, samples, *args, **kwargs):
        super(KdePdf, self).__init__(*args, **kwargs)
        self.kde = stats.gaussian_kde(samples)
        low = min(samples)
        high = max(samples)
        self._xgrid = np.linspace(low, high, NPTS + 1)

    def density(self, xs):
        return self.kde.evaluate(xs)

    def xgrid(self):
        return self._xgrid


class Pdfs:
    """
    Plot a sequence of Pdf instances.
    """

    def __init__(self, pdfs):
        self.pdfs = pdfs

    def plot(self):
        # labels = []
        for p in self.pdfs:
            p.plot()
        plt.legend()


class Gaussian(Pdf):
    """
    Plot a Gaussian PDF
    """

    def __init__(self, mean=0.0, std=1.0, *args, **kwargs):
        super(Gaussian, self).__init__(*args, **kwargs)
        self.mean = mean
        self.std = std
        self.dist = stats.norm(loc=mean, scale=std)

    def density(self, xs):
        return self.dist.pdf(xs)

    def xgrid(self):
        _min = (self.mean - 4 * self.std,)
        _max = self.mean + 4 * self.std
        return np.linspace(_min, _max, NPTS + 1)


class Pmf(Plottable):
    """
    Base class for plotting probability mass functions.
    """

    def density(self, xs):
        raise NotImplementedError("Implement Me")

    def xgrid(self):
        """
        Return the default x-values for plotting
        """
        raise NotImplementedError("Implement Me")

    def get_series(self):
        xs = self.xgrid()
        ys = self.density(xs)
        return xs, ys

    def plot(self, plot_type="step", **plot_args):
        """
        Parameters
        ---------
        plot_type: str
            The type of plot mode to use, can one of matplotlib's default plot
            types (e.g. 'bar', 'plot', 'scatter')
        """
        (
            xs,
            ys,
        ) = self.get_series()
        plotfun = getattr(plt, plot_type)
        plotfun(xs, ys, label=self.label, color=self.color, **plot_args)

    def sample(self, size):
        return self.dist.rvs(size=size)


class Binomial(Pmf):
    """
    Plot a Binomial PMF
    """

    def __init__(self, n=20, p=0.5, *args, **kwargs):
        super(Binomial, self).__init__(*args, **kwargs)
        self.n = n
        self.p = p
        self.dist = stats.binom(n, p)

    def density(self, xs):
        return self.dist.pmf(xs)

    def xgrid(self):
        return np.arange(0, self.n)


class Bernoulli(Pmf):
    """
    Plot a Bernoulli PDF
    """

    def __init__(self, plot_type="bar", p=0.5, *args, **kwargs):
        super(Bernoulli, self).__init__(*args, **kwargs)
        self.plot_type = plot_type
        self.p = p
        self.dist = stats.bernoulli(p)

    def density(self, xs):
        return self.dist.pmf(xs)

    def xgrid(self):
        return np.linspace(0.0, 1.0, 2)


class Poisson(Pmf):
    """
    Plot a Binomial PMF
    """

    def __init__(self, mu=1, *args, **kwargs):
        super(Poisson, self).__init__(*args, **kwargs)
        self.mu = mu
        self.dist = stats.poisson(mu)

    def density(self, xs):
        return self.dist.pmf(xs)

    def xgrid(self):
        return np.arange(0, max([1 + self.mu * 2.0, 11]))


def plot_interval(
    left,
    right,
    middle,
    color=None,
    display_text=False,
    label=None,
    y=0.0,
    offset=0.005,
    fontsize=14,
):
    color = color if color else "k"
    text_y = y + offset

    if middle in (-np.inf, np.inf) and (
        left in (np.inf, -np.inf) or right in (np.inf, -np.inf)
    ):
        raise ValueError("too many interval values are inf")

    _left = middle - 4 * np.abs(right) if left in (np.inf, -np.inf) else left
    _right = middle + 4 * np.abs(left) if right in (np.inf, -np.inf) else right

    plt.plot((_left, _right), (y, y), color=color, linewidth=3, label=label)
    plt.plot(middle, y, "o", color=color, markersize=10)

    if display_text:
        label = "{}\n({}, {})".format(round(middle, 2), round(left, 2), round(right, 2))
        plt.text(middle, text_y, label, ha="center", fontsize=fontsize, color=color)


def raise_y(ax, baseline=0):
    ylims = ax.get_ylim()
    ax.set_ylim(baseline, ylims[1])
    return ax


def lower_y(ax, baseline=None):
    ylims = ax.get_ylim()

    baseline = baseline if baseline else ylims[0] - np.abs(ylims[1]) * 0.05

    ax.set_ylim(baseline, ylims[1])
    return ax


def save_figure(post_name: str, fig_name: str):
    """Save figure in the current notebook cell"""
    fig_dir = os.path.join(FIGS_DIR, post_name)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fig_file = os.path.join(fig_dir, fig_name + ".png")
    plt.savefig(fig_file, dpi=300)
