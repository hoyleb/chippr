# Module containing generally handy functions used for plotting

lnz, nz = r'$\ln[n(z)]$', r'$n(z)$'

s_tru, w_tru, a_tru, c_tru, d_tru, l_tru = '--', 0.5, 1., 'k', [(0, (1, 1))], 'True '
s_int, w_int, a_int, c_int, d_int, l_int = '--', 0.5, 0.5, 'k', [(0, (1, 1))], 'Interim '
s_stk, w_stk, a_stk, c_stk, d_stk, l_stk = '--', 1.5, 1., 'k', [(0, (7.5, 7.5))], 'Stacked '
s_map, w_map, a_map, c_map, d_map, l_map = '--', 1., 1., 'k', [(0, (7.5, 7.5))], 'MMAP '
s_exp, w_exp, a_exp, c_exp, d_exp, l_exp = '--', 1., 1., 'k', [(0, (2.5, 2.5))], 'MExp '
s_mle, w_mle, a_mle, c_mle, d_mle, l_mle = '--', 2., 1., 'k', [(0, (2.5, 2.5))], 'MMLE '
# s_smp, w_smp, a_smp, c_smp, d_smp, l_smp = '--', 1., 1., 'k', [(0, (1, 1))], 'Sampled '
s_bfe, w_bfe, a_bfe, c_bfe, d_bfe, l_bfe = '--', 2., 1., 'k', [(0, (1, 1))], 'Mean of\n Samples '

def plot_step(sub_plot, bin_ends, plot, s='--', c='k', a=1, w=1, d=[(0,(1,0.0001))], l=None, r=False):
    """
    Plots a step function

    Parameters
    ----------
    sub_plot: matplotlib.pyplot subplot object
        subplot into which step function is drawn
    bin_ends: list or ndarray
        list or array of endpoints of bins
    plot: list or ndarray
        list or array of values within each bin
    s: string, optional
        matplotlib.pyplot linestyle
    c: string, optional
        matplotlib.pyplot color
    a: int or float, [0., 1.], optional
        matplotlib.pyplot alpha (transparency)
    w: int or float, optional
        matplotlib.pyplot linewidth
    d: list of tuple, optional
        matplotlib.pyplot dash style, of form [(start_point, (points_on, points_off, ...))]
    l: string, optional
        label for function
    r: boolean, optional
        True for rasterized, False for vectorized
    """

    plot_h(sub_plot, bin_ends, plot, s, c, a, w, d, l, r)
    plot_v(sub_plot, bin_ends, plot, s, c, a, w, d, r)

def plot_h(sub_plot, bin_ends,plot, s='--', c='k', a=1, w=1, d=[(0,(1,0.0001))], l=None, r=False):
    """
    Helper function to plot horizontal lines of a step function

    Parameters
    ----------
    sub_plot: matplotlib.pyplot subplot object
        subplot into which step function is drawn
    bin_ends: list or ndarray
        list or array of endpoints of bins
    plot: list or ndarray
        list or array of values within each bin
    s: string, optional
        matplotlib.pyplot linestyle
    c: string, optional
        matplotlib.pyplot color
    a: int or float, [0., 1.], optional
        matplotlib.pyplot alpha (transparency)
    w: int or float, optional
        matplotlib.pyplot linewidth
    d: list of tuple, optional
        matplotlib.pyplot dash style, of form [(start_point, (points_on, points_off, ...))]
    l: string, optional
        label for function
    r: boolean, optional
        True for rasterized, False for vectorized
    """

    sub_plot.hlines(plot,
                   bin_ends[:-1],
                   bin_ends[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a,
                   label=l,
                   rasterized=r)
def plot_v(sub_plot, bin_ends, plot, s='--', c='k', a=1, w=1, d=[(0,(1,0.0001))], r=False):
    """
    Helper function to plot vertical lines of a step function

    Parameters
    ----------
    sub_plot: matplotlib.pyplot subplot object
        subplot into which step function is drawn
    bin_ends: list or ndarray
        list or array of endpoints of bins
    plot: list or ndarray
        list or array of values within each bin
    s: string, optional
        matplotlib.pyplot linestyle
    c: string, optional
        matplotlib.pyplot color
    a: int or float, [0., 1.], optional
        matplotlib.pyplot alpha (transparency)
    w: int or float, optional
        matplotlib.pyplot linewidth
    d: list of tuple, optional
        matplotlib.pyplot dash style, of form [(start_point, (points_on, points_off, ...))]
    r: boolean, optional
        True for rasterized, False for vectorized
    """

    sub_plot.vlines(bin_ends[1:-1],
                   plot[:-1],
                   plot[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a,
                   rasterized=r)
