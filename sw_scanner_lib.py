import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.distance import jensenshannon
# import pandas as pd

def round_up_to_minute(datetime):
    # Round up to the nearest minute
    rounded_datetime = ((datetime.astype('datetime64[ns]') + np.timedelta64(1, 'm') - np.timedelta64(1, 'ns')).astype('datetime64[m]')).astype('datetime64[ns]')
    return rounded_datetime


def round_down_to_minute(datetime):
    # Round down to the nearest minute
    rounded_datetime = datetime.astype('datetime64[m]').astype('datetime64[ns]')
    return rounded_datetime

def f(x,a,b):
    return a*x+b

def js_divergence(btot1, n_sigma, nbins):
    # discard points outside of n sigma
    # solar wind has a much higher chance than gaussian to have extreme values
    mean = np.nanmean(btot1)
    std = np.nanstd(btot1)
    keep_ind = (btot1 > mean - n_sigma*std) & (btot1 < mean + n_sigma*std)
    btot1[np.invert(keep_ind)] = np.nan
    nan_ratio = np.sum(np.isnan(btot1))/len(btot1)
    x = btot1[np.invert(np.isnan(btot1))]

    # rescale x
    x = (x-np.mean(x))/np.std(x)

    # calculate pdf of x
    # nbins = divergence['js']['nbins']
    bins = np.linspace(-n_sigma, n_sigma, nbins)
    hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)

    # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
    bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
    pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)

    js_div = jensenshannon(hist_data, pdf_gaussian)

    return js_div, nan_ratio


def calc_xinds(index, t_step, use_pandas=True):

    print("t step: %s" % t_step)

    if use_pandas:
        import pandas as pd
        tgrid0 = pd.Timestamp(index[0]).ceil('1min')
        nxgrid = (index[-1] - tgrid0)/t_step
        xgrid = np.arange(tgrid0, index[-1], t_step)
        print("length of xgrid: %d, tstep=%s, range: %s -> %s" %(len(xgrid), pd.Timedelta(t_step), pd.Timestamp(xgrid[0]), pd.Timestamp(xgrid[-1])))
    else:
        tgrid0 = round_up_to_minute(index[0])
        nxgrid = (index[-1] - tgrid0)/t_step
        xgrid = np.arange(tgrid0, index[-1], t_step)
        print("Length of xgrid: %d, tstep=%s, range: %s -> %s" %(len(xgrid), t_step/np.timedelta64(1,'s'), xgrid[0], xgrid[-1]))

    xinds = np.zeros(len(xgrid), dtype = int)
    xind_last = int(0)
    jumpsize = int(5*len(index)/((index[-1]-index[0])/np.timedelta64(1,'s'))*t_step/np.timedelta64(1,'s'))
    for i1, xg in enumerate(xgrid):
        xg = xgrid[i1]
        try:
            xind_diff = int(np.where(index[xind_last:xind_last+jumpsize] < xg)[0][-1])
        except:
            xind_diff = int(np.where(index[xind_last:] < xg)[0][-1])
        xind_new = xind_diff+xind_last
        xinds[i1] = xind_new
        xind_last = xind_new

    xinds_infos = {
        'xinds': xinds,
        'xgrid': xgrid,
        'nxgrid': nxgrid,
        'jumpsize': jumpsize
    }

    return xinds_infos