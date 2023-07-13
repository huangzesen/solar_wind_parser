import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.distance import jensenshannon
import pickle
import tqdm
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


def SolarWindScanner(
    index,
    Btot,
    Dist_au,
    settings = None,
    verbose = False,
    Ncores = 8,
    use_pandas = True,
    progress = True,
    temporary_path = None
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

    if progress:
        import tqdm

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

        print("Saving Temporary files to %s"%(temporary_path))

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

    # # discard unphysical values
    # ind = Btot < 0.1
    # Btot[ind] = np.nan
    # print("%d out of %d unphysical! Ratio %.4f %%" %(np.sum(ind), len(Btot), np.sum(ind)/len(Btot)*100))

    print("Mode: parallel")

    alloc_input = {
        'index': index,
        'Btot': Btot,
        'Dist_au': Dist_au,
        'settings': settings,
        'tranges': tranges
    }

    if temporary_path is not None:
        if progress:
            with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
                savpoint = N // 5  # Calculate what 10% of N is
                for i, scan in enumerate(tqdm.tqdm(p.imap_unordered(SolarWindScannerInnerLoopParallel, range(N)), total=N, mininterval=5)):
                    scans.append(scan)
                    if (i+1) % savpoint == 0:  # If the current iteration is a multiple of 10% of N
                        with open(temporary_path+'/'+f'scans_{((i+1)//savpoint)*20}percent.pkl', 'wb') as f:
                            pickle.dump(scans, f)
        else:
            with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
                savpoint = N // 5  # Calculate what 10% of N is
                for i, scan in enumerate(p.imap_unordered(SolarWindScannerInnerLoopParallel, total=N)):
                    scans.append(scan)
                    if (i+1) % savpoint == 0:  # If the current iteration is a multiple of 10% of N
                        with open(temporary_path+'/'+f'scans_{((i+1)//savpoint)*20}percent.pkl', 'wb') as f:
                            pickle.dump(scans, f)

    
    if progress:
        with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
            for scan in tqdm.tqdm(p.imap_unordered(SolarWindScannerInnerLoopParallel, range(N)), total=N, mininterval=5):
                scans.append(scan)
    else:
        with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
            for scan in p.imap_unordered(SolarWindScannerInnerLoopParallel, range(N)):
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




