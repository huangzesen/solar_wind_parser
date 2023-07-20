import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.distance import jensenshannon
import pickle
import tqdm
# import pandas as pd
from sw_scanner_lib import round_up_to_minute, round_down_to_minute,f,js_divergence
from gc import collect


def SolarWindScanner(
    Btot,
    Dist_au,
    settings = None,
    verbose = False,
    Ncores = 8,
    use_pandas = True,
    chunksize = 100,
    mininterval = 10,
    collect_garbage = True
):

    """
    Solar Wind Scanner
    Input:
        index: numpy array of np.datetime64
        Btot: numpy array of float
        Dist_au: numpy array of float
        settings: dictionary
            wins: numpy array of np.timedelta64
            step: np.timedelta64
            normality_mode: str
                divergence: calculate the Jensen-Shannon Divergence
                find_nan: calculate the nan ratio
            
    """

    if use_pandas:
        import pandas as pd

    if verbose:
        print("\n-----------------------------------------------------")
        print("       Solar Wind Scanner Super Computer Xinds")
        print("-----------------------------------------------------\n")
        print("---- Settings ----\n")
        for k,v in settings.items():
            if k == 'wins':
                if use_pandas:
                    print("%s %s --> %s"%(k, pd.Timedelta(np.min(v)), pd.Timedelta(np.max(v))))
                else:
                    print("%s %s --> %s"%(k, np.min(v), np.max(v)))
            else:
                if use_pandas:
                    if type(v) is np.timedelta64:
                        print(k, pd.Timedelta(v))
                    else:
                        print(k, v)
                else:
                    print(k,v)

        print("\n---- End of settings ----\n\n")

    wins = settings['wins']
    step = settings['step']
    xgrid = settings['xgrid']

    N = 0
    
    win_list = []
    for i1, win in enumerate(wins):
        tstart0 = xgrid[0]
        tend0 = tstart0 + win
        tends = np.arange(tend0, xgrid[-1], step)
        N1 = len(tends)
        N = N+N1
        for j1 in range(N1):
            win_list.append({'nsteps': j1, 'win': win})


    # print the input value range and Btot index range
    print('xgrid: %s - %s' %(xgrid[0], xgrid[-1]))

    print("\nTotal intervals= %d\n" %(N))

    alloc_input = {
        'Btot': Btot,
        'Dist_au': Dist_au,
        'settings': settings,
        'win_list': win_list
    }


    scans = []

    settings['collect_garbage'] = collect_garbage
    print("Current Chunk size: %d" % chunksize)
    print("Garbage collection: %s" % collect_garbage)

    with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
        for scan in tqdm.tqdm(p.imap_unordered(SolarWindScannerInnerLoopParallel, range(N), chunksize), total=N, mininterval=mininterval):
            scans.append(scan)


    return scans


def InitParallelAllocation(alloc_input):
    global Btot, Dist_au, settings, win_list
    Btot = alloc_input['Btot']
    Dist_au = alloc_input['Dist_au']
    settings = alloc_input['settings']
    win_list = alloc_input['win_list']



def SolarWindScannerInnerLoopParallel(i1):
    # access global variables
    global Btot, Dist_au, settings, win_list

    n_sigma = settings['n_sigma']
    normality_mode = settings['normality_mode']
    step = settings['step']
    xinds = settings['xinds']
    xgrid = settings['xgrid']
    capsize = settings['capsize']

    collect_garbage = settings['collect_garbage']

    win = win_list[i1]['win']
    nsteps = win_list[i1]['nsteps']

    tstart0 = xgrid[0]
    tstart= tstart0 + nsteps * step
    tend = tstart + win
    id0 = xinds[int(np.where(xgrid == tstart)[0][0])]
    id1 = xinds[int(np.where(xgrid == tend)[0][0])]

    skip_size = int(np.floor((id1-id0)/capsize))
    if skip_size == 0:
        skip_size = 1
    # btot = np.copy(Btot[id0:id1:skip_size])
    btot = Btot[id0:id1:skip_size]

    nan_infos = {
        # 'raw': {'len': len(btot), 'count': np.sum(np.isnan(btot)),'ratio': np.sum(np.isnan(btot))/len(btot)},
        'ids': {'id0': id0, 'id1': id1, 'skip_size': skip_size, 'len': len(btot)}
    }
    nan_infos['flag'] = 1

    if normality_mode == 'divergence':
        # r = np.copy(Dist_au[id0:id1:skip_size])
        r = Dist_au[id0:id1:skip_size]
        divergence = settings['divergence']

        try:

            # find the rescaling scale with r
            rfit = curve_fit(f, np.log10(r), np.log10(btot))
            scale = -rfit[0][0]

            # normalize btot with r
            btot1 = btot * ((r/r[0])**scale)

            # dist ratio
            r_ratio = np.max(r)/np.min(r)
            
            x = btot1

            # rescale x
            x = (x-np.mean(x))/np.std(x)

            # calculate the distance
            distances = {}

            # calculate pdf of x
            nbins = divergence['js']['nbins']
            bins = np.linspace(-n_sigma, n_sigma, nbins)
            hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)
            outside_count = np.sum(x < bins[0]) + np.sum(x > bins[-1])


            # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
            bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
            pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)

            js_div = jensenshannon(hist_data, pdf_gaussian)
            distances['js'] = js_div

            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'distances': distances,
                'r_ratio': r_ratio,
                'nan_infos': nan_infos,
                'fit_results': rfit,
                'histogram_infos': {
                    # 'hist_data': hist_data, 
                    # 'bin_edges_data': bin_edges_data,
                    'n_sigma': n_sigma,
                    'outside_count': outside_count
                    }
            }


        except:
            # raise ValueError("fuck")
            rfit = (
                np.array([np.nan,np.nan]), np.array([[np.nan,np.nan],[np.nan,np.nan]])
            )
            distances = {'js':np.nan}
            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'distances': distances,
                'nan_infos': nan_infos,
                'fit_results': rfit,
                'histogram_infos': {
                    'n_sigma': n_sigma,
                    'outside_count': np.nan
                    }
            }


    elif normality_mode == 'divergence_scaled_btot':

        divergence = settings['divergence']

        try:

            x = np.copy(btot)

            # rescale x
            x = (x-np.mean(x))/np.std(x)

            # calculate the distance
            distances = {}

            # calculate pdf of x
            nbins = divergence['js']['nbins']
            bins = np.linspace(-n_sigma, n_sigma, nbins)
            hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)
            outside_count = np.sum(x < bins[0]) + np.sum(x > bins[-1])


            # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
            bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
            pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)

            js_div = jensenshannon(hist_data, pdf_gaussian)
            distances['js'] = js_div

            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'distances': distances,
                'nan_infos': nan_infos,
                'histogram_infos': {
                    'n_sigma': n_sigma,
                    'outside_count': outside_count
                    }
            }


        except:
            # raise ValueError("fuck")
            distances = {'js':np.nan}
            scan = {
                't0': tstart,
                't1': tend,
                'nan_ratio': np.nan,
                'distances': distances,
                'nan_infos': nan_infos,
                'histogram_infos': {
                    'n_sigma': n_sigma,
                    'outside_count': np.nan
                    }
            }


    elif normality_mode == 'find_nan':

        scan = {
            't0': tstart,
            't1': tend,
            'win': win,
            'nan_infos': nan_infos
        }


    else:
        raise ValueError("Wrong mode: %s" %(normality_mode))

    if collect_garbage:
        if i1 % 10 == 0:
            collect()

    return scan





