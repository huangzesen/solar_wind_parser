import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from scipy.optimize import curve_fit
import tqdm
from scipy import stats
from scipy.spatial.distance import jensenshannon
from gc import collect
# import pandas as pd




def SolarWindScanner(
    Btot,
    Dist_au,
    settings = None,
    verbose = False,
    Ncores = 8,
    use_pandas = True,
    chunksize = 100
):

    """
    Solar Wind Scanner
    Input:
    """

    if 'normality_mode' not in settings.keys():
        settings['normality_mode'] = 'divergence'

    if use_pandas:
        import pandas as pd

    if verbose:
        print("\n--------------------------------------------")
        print("\tSolar Wind Scanner Pro Xinds")
        print("--------------------------------------------\n")
        print("---- Settings ----\n")
        for k,v in settings.items():
            if use_pandas:
                if type(v) is np.timedelta64:
                    print(k, pd.Timedelta(v))
                else:
                    print(k,v)
            else:
                print(k,v)

        print("\n---- End of settings ----\n\n")

    win = settings['win']
    step = settings['step']
    xgrid = settings['xgrid']

    # print the input value range and Btot index range
    print('xgrid: %s - %s' %(xgrid[0], xgrid[-1]))

    tstart0 = xgrid[0]
    tend0 = tstart0 + win
    tends = np.arange(tend0, xgrid[-1], step)
    N = len(tends)
    

    scans = []



    alloc_input = {
        'Btot': Btot,
        'Dist_au': Dist_au,
        'settings': settings
    }
    
    with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
        for scan in tqdm.tqdm(p.imap_unordered(SolarWindScannerInnerLoopParallel, range(N), chunksize), total=N):
            scans.append(scan)



    return scans


def InitParallelAllocation(alloc_input):
    global Btot, Dist_au, settings
    Btot = alloc_input['Btot']
    Dist_au = alloc_input['Dist_au']
    settings = alloc_input['settings']



def SolarWindScannerInnerLoopParallel(i1):
    # access global variables
    global Btot, Dist_au, settings

    win = settings['win']
    step = settings['step']
    xinds = settings['xinds']
    xgrid = settings['xgrid']
    n_sigma = settings['n_sigma']
    normality_mode = settings['normality_mode']

    tstart0 = xgrid[0]

    tstart= tstart0 + i1 * step
    tend = tstart + win
    id0 = xinds[int(np.where(xgrid == tstart)[0][0])]
    id1 = xinds[int(np.where(xgrid == tend)[0][0])]

    btot = Btot[id0:id1]
    nan_infos = {
        'raw': {'len': len(btot), 'count': np.sum(np.isnan(btot)),'ratio': np.sum(np.isnan(btot))/len(btot)},
        'ids': {'id0': id0, 'id1': id1}
    }
    nan_infos['flag'] = 1

    if normality_mode == 'divergence':
        r = Dist_au[ind]
        divergence = settings['divergence']

        try:

            # find the rescaling scale with r
            ind = np.invert((np.isnan(r)) | np.isnan(btot))
            rfit = curve_fit(f, np.log10(r[ind]), np.log10(btot[ind]))
            scale = -rfit[0][0]

            # normalize btot with r
            btot1 = btot * ((r/r[0])**scale)

            # dist ratio
            r_ratio = np.max(r)/np.min(r)
            
            
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

            # calculate the distance
            distances = {}

            # calculate pdf of x
            nbins = divergence['js']['nbins']
            bins = np.linspace(-n_sigma, n_sigma, nbins)
            hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)

            # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
            bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
            pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)

            js_div = jensenshannon(hist_data, pdf_gaussian)
            distances['js'] = js_div

            scan = {
                't0': tstart,
                't1': tend,
                'nan_ratio': nan_ratio,
                'distances': distances,
                'r_ratio': r_ratio,
                'settings': settings,
                'fit_results': rfit,
                'nan_infos': nan_infos
            }


        except:
            rfit = (
                np.array([np.nan,np.nan]), np.array([[np.nan,np.nan],[np.nan,np.nan]])
            )
            distances = {'js':np.nan}
            scan = {
                't0': tstart,
                't1': tend,
                'nan_ratio': np.nan,
                'distances': distances,
                'r_ratio': np.nan,
                'settings': settings,
                'fit_results': rfit,
                'nan_infos': nan_infos
            }

    elif normality_mode == 'divergence_scaled_btot':

        divergence = settings['divergence']

        try:

            btot1 = btot
            
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

            # calculate the distance
            distances = {}

            # calculate pdf of x
            nbins = divergence['js']['nbins']
            bins = np.linspace(-n_sigma, n_sigma, nbins)
            hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)

            # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
            bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
            pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)

            js_div = jensenshannon(hist_data, pdf_gaussian)
            distances['js'] = js_div

            scan = {
                't0': tstart,
                't1': tend,
                'nan_ratio': nan_ratio,
                'distances': distances,
                'settings': settings,
                'nan_infos': nan_infos
            }

            del locals()['btot']
            del locals()['x']
            del locals()['keep_ind']
            del locals()['btot1']
            del nbins, bins, hist_data, bin_edges_data, bin_midpoints, pdf_gaussian
            collect()


        except:
            raise ValueError("fuck")
            distances = {'js':np.nan}
            scan = {
                't0': tstart,
                't1': tend,
                'nan_ratio': np.nan,
                'distances': distances,
                'settings': settings,
                'nan_infos': nan_infos
            }


    elif normality_mode == 'find_nan':

        scan = {
            't0': tstart,
            't1': tend,
            'settings': settings,
            'nan_infos': nan_infos
        }


    else:
        raise ValueError("Wrong mode: %s" %(normality_mode))

    collect()
    return scan



def f(x,a,b):
    return a*x+b



