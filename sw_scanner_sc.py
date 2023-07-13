import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.distance import jensenshannon
import pickle
import tqdm
# import pandas as pd
from sw_scanner_lib import round_up_to_minute, round_down_to_minute,f,js_divergence


def SolarWindScanner(
    index,
    Btot,
    Dist_au,
    settings = None,
    verbose = False,
    Ncores = 8,
    use_pandas = True
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
        print("\n-----------------------------------------------")
        print("       Solar Wind Scanner Super Computer")
        print("-----------------------------------------------\n")
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
    
    sample_cadence = index[1]-index[0]
    if use_pandas:
        sample_cadence = pd.Timedelta(sample_cadence)
    print("Sampling cadence: %s" %(sample_cadence))

    N = 0
    tranges = []
    tstart0 = round_down_to_minute(index[0])
    tendmax = tstart0
    for i1, win in enumerate(wins):
        tend0 = tstart0 + win
        scans = []

        # calculate num of steps
        N1 = int(np.floor((index[-1] - tend0)/step)) + 1

        for j1 in range(N1):
            tstart= tstart0 + j1 * step
            tend = tstart + win

            tranges.append({'tstart': tstart, 'tend': tend, 'win': win})

            tendmax = np.max([tendmax, tend])

        N = N + N1

    # print the input value range and Btot index range
    print('Index: %s - %s' %(index[0], index[-1]))
    print('Final Range: %s - %s' %(tstart0, tendmax))

    print("\nTotal intervals= %d\n" %(N))

    alloc_input = {
        'index': index,
        'Btot': Btot,
        'Dist_au': Dist_au,
        'settings': settings,
        'tranges': tranges
    }


    with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
        for scan in tqdm.tqdm(p.imap_unordered(SolarWindScannerInnerLoopParallel, range(N)), total=N, mininterval=0.1):
            scans.append(scan)


    return scans


def InitParallelAllocation(alloc_input):
    global index, Btot, Dist_au, tranges, settings
    index = alloc_input['index']
    Btot = alloc_input['Btot']
    Dist_au = alloc_input['Dist_au']
    settings = alloc_input['settings']
    tranges = alloc_input['tranges']



def SolarWindScannerInnerLoopParallel(i1):
    # access global variables
    global index, Btot, Dist_au, settings, tranges

    n_sigma = settings['n_sigma']
    normality_mode = settings['normality_mode']

    tstart = tranges[i1]['tstart']
    win = tranges[i1]['win']
    tend = tranges[i1]['tend']
    ind = (index >= tstart) & (index < tend)
    btot = Btot[ind]
    r = Dist_au[ind]

    nan_infos = {
        'raw': {'len': len(btot), 'count': np.sum(np.isnan(btot)),'ratio': np.sum(np.isnan(btot))/len(btot)}
    }
    nan_infos['flag'] = 1

    if normality_mode == 'divergence':
        divergence = settings['divergence']

        try:

            # calculate the distance
            distances = {}
            
            # usual divergence
            js_div, nan_ratio = js_divergence(btot, n_sigma, divergence['js']['nbins'])
            distances['js'] = {'divergence': js_div, 'nan_ratio': nan_ratio}

            # log divergence
            js_div, nan_ratio = js_divergence(np.log10(btot), n_sigma, divergence['js']['nbins'])
            distances['js_log'] = {'divergence': js_div, 'nan_ratio': nan_ratio}


            # find the rescaling scale with r
            ind = np.invert((np.isnan(r)) | np.isnan(btot))
            rfit = curve_fit(f, np.log10(r[ind]), np.log10(btot[ind]))
            scale = -rfit[0][0]

            # normalize btot with r
            btot1 = btot * ((r/r[0])**scale)

            # dist ratio
            r_ratio = np.max(r)/np.min(r)

            js_div, nan_ratio = js_divergence(btot1, n_sigma, divergence['js']['nbins'])
            distances['js_r_scaled'] = {'divergence': js_div, 'nan_ratio': nan_ratio}

            js_div, nan_ratio = js_divergence(np.log10(btot1), n_sigma, divergence['js']['nbins'])
            distances['js_log_r_scaled'] = {'divergence': js_div, 'nan_ratio': nan_ratio}

            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                # 'nan_ratio': nan_ratio,
                'distances': distances,
                'r_ratio': r_ratio,
                'fit_results': rfit,
                'nan_infos': nan_infos
            }


        except:
            rfit = (
                np.array([np.nan,np.nan]), np.array([[np.nan,np.nan],[np.nan,np.nan]])
            )
            distances = None
            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'distances': distances,
                'r_ratio': np.nan,
                'fit_results': rfit,
                'nan_infos': nan_infos
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

    return scan





