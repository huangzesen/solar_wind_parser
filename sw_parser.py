import pandas as pd
import numpy as np
import time, sys
from scipy.stats import shapiro, kstest
from scipy.interpolate import interp1d
from multiprocessing import Pool
from scipy.optimize import curve_fit
import tqdm
from gc import collect
from parfor import parfor

# default settings
default_settings = {
    # initial moving window size
    'win': pd.Timedelta('40min'),

    # outer loop step size
    'step_out': pd.Timedelta('1min'),

    # inner loop step size
    'step_in': pd.Timedelta('1min'),

    # thresh to accept
    'normality_thresh_in': 0.4,

    # thresh to stop
    'normality_thresh_out': 0.5,

    # thresh to keep
    'normality_thresh_keep': 0.5,

    # thresh to keep for the good intervals
    # if this interval has normality > 0.5, drop up till 0.5
    # 0.5 is very good gaussian distribution!!
    'normality_thresh_good': 0.8,

    # for good interval, raise the exit thresh
    'normality_thresh_good_out': 0.4,

    # fast forward size if current interval is all pop out, 
    # measured in terms of step_out
    'forward_size': 5,

    # fall back size after devouring
    # measured in terms of step_out
    # Note that if current interval is good, there will be no fallback
    'fallback_size': 10
}



def SolarWindScanner(
    Btot,
    Dist_au,
    settings = None,
    parallel = False,
    verbose = False,
    Ncores = 8
):

    """
    Solar Wind Scanner
    Input:
    """

    if 'methods' not in settings.keys():
        settings['methods'] = {
            'shapiro': {'Ntests': 2000},
            'kstest': {'Ntests': 600}
        }
    
    if 'downsample_size' not in settings.keys():
        settings['downsample_size'] = 500

    if verbose:
        print("\n----------------------------------")
        print("\tSolar Wind Scanner")
        print("----------------------------------\n")
        print("Settings: ")
        for k,v in settings.items():
            print(k,v)

    win = settings['win']
    step = settings['step']

    tstart0 = Btot.index[0]
    tend0 = tstart0 + win
    scans = []

    # calculate num of steps
    N = int(np.floor((Btot.index[-1] - tend0)/step)) + 1
    inputs = []

    print('Interpolating Dist_au...')

    # interpolate Dist_au with scipy.interpolate.interp1d
    ts = np.array([t.timestamp() for t in Dist_au.index])
    f_dist_au = interp1d(ts, Dist_au.values, 'linear', fill_value='extrapolate')

    # create Btot unix index
    Btot_index_unix = np.array([t.timestamp() for t in Btot.index])

    # print the input value range and Btot index range
    print('Dist_au: %s - %s' %(Dist_au.index[0], Dist_au.index[-1]))
    print('Btot: %s - %s' %(Btot.index[0], Btot.index[-1]))

    print("\nTotal intervals= %d\n" %(N))

    if parallel:
        print("Mode: parallel")

        alloc_input = {
            'Btot': Btot,
            # 'Dist_au': Dist_au,
            'f_dist_au': f_dist_au,
            'Btot_index_unix': Btot_index_unix,
            'tstart0': tstart0,
            'settings': settings
        }
        
        with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
            for scan in tqdm.tqdm(p.imap_unordered(SolarWindScannerInnerLoopParallel, range(N)), total=N):
                scans.append(scan)


    else:

        print("Mode: sequential")
        for i1 in range(N):
            tstart= tstart0 + i1 * step
            tend = tstart + win
            ind = (Btot.index >= tstart) & (Btot.index < tend)
            btot = Btot[ind]
            r = Dist_au[ind]
            input = {
                'tstart': tstart,
                'tend': tend,
                'btot': btot,
                'r': r
            }
            inputs.append(input)

            print("%d out of %d allocated, %.2f%% finished" %(i1, N, i1/N*100), end='\r')
        
        print("Finished! Occupying %.2f MB, Start scanning...\n" %(sys.getsizeof(inputs)/1024.))

        # print("Scanning Mode: Sequential")
        for i1 in range(N):
            scan = SolarWindScannerInnerLoopSequential(inputs[i1])
            scans.append(scan)

            if verbose:
                print(r"scanning...: tstart=%s, tend=%s, normality=%.4f, finished = %.2f%%" %(
                    scan['t0'], 
                    scan['t1'], 
                    scan['normality'], 
                    (scan['t1']-Btot.index[0])/(Btot.index[-1]-Btot.index[0])*100
                ), end='\r')

    return scans


def InitParallelAllocation(alloc_input):
    global Btot, tstart0, settings, f_dist_au, Btot_index_unix
    Btot = alloc_input['Btot'].copy()
    # Dist_au = alloc_input['Dist_au'].copy()
    tstart0 = alloc_input['tstart0']
    settings = alloc_input['settings']
    f_dist_au = alloc_input['f_dist_au']
    Btot_index_unix = alloc_input['Btot_index_unix']


def SolarWindScannerInnerLoopParallel(i1):
    # access global variables
    global Btot, tstart0, settings, f_dist_au, Btot_index_unix

    win = settings['win']
    step = settings['step']
    methods = settings['methods']
    downsample_size = settings['downsample_size']
    n_sigma = settings['n_sigma']

    tstart= tstart0 + i1 * step
    tend = tstart + win
    ind = (Btot.index >= tstart) & (Btot.index < tend)
    indices = Btot.index[ind]
    btot = Btot[indices].values

    try:

        ts = Btot_index_unix[ind]
        r = f_dist_au(ts)

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
        mean = np.mean(btot1)
        std = np.std(btot1)
        keep_ind = (btot1 > mean - n_sigma*std) & (btot1 < mean + n_sigma*std)
        btot1[np.invert(keep_ind)] = np.nan
        nan_ratio = np.sum(np.isnan(btot1))/len(btot1)
        x = btot1[np.invert(np.isnan(btot1))]

        normalities = {}    
        for method, ms in methods.items():
            if len(x) < 500:
                # if not enough samples, continue
                normality = np.nan
            else:
                # normalize input for normality test
                x = (x-np.mean(x))/np.std(x)

                # normality test
                if method == 'shapiro':
                    a1 = np.array([shapiro(np.random.choice(x, size=downsample_size, replace = False)).pvalue for i1 in range(ms['Ntests'])])
                elif method == 'kstest':
                    # normalize the distribution for kstest
                    a1 = np.array([kstest(np.random.choice(x, size=downsample_size, replace = False), 'norm').pvalue for i1 in range(ms['Ntests'])])
                else:
                    raise ValueError("Wrong method")
                normality = np.sum(a1 > 0.05)/len(a1)
            
            normalities[method] = normality
        
        scan = {
            't0': tstart,
            't1': tend,
            'nan_ratio': nan_ratio,
            'normalities': normalities,
            'r_ratio': r_ratio,
            'settings': settings,
            'fit_results': rfit
        }

    except:
        r_ratio = np.max(r)/np.min(r)
        normalities = {}    
        for method, ms in methods.items():
            normalities[method]=np.nan
            
        rfit = (
            np.array([np.nan,np.nan]), np.array([[np.nan,np.nan],[np.nan,np.nan]])
        )
        scan = {
            't0': tstart,
            't1': tend,
            'nan_ratio': 1.0,
            'normalities': normalities,
            'r_ratio': r_ratio,
            'settings': settings,
            'fit_results': rfit
        }

    return scan



def f(x,a,b):
    return a*x+b



def SolarWindParser(
        Btot, 
        Dist_au,
        df_scans,
        settings=None,
        scan_settings = None, 
        verbose=True
    ):
    """
    Solar Wind Parser
    Input:
    """
    tl0 = time.time()

    # settings
    if settings is None:
        settings = default_settings

    if verbose:
        print("\n----------------------------------")
        print("\tSolar Wind Parser")
        print("----------------------------------\n")
        print("Settings: ")
        for k,v in settings.items():
            print(k,v)
        print("Scan settings: ")
        for k,v in scan_settings.items():
            print(k,v)

    win = scan_settings['win']
    step_out = scan_settings['step']
    methods = scan_settings['methods']
    downsample_size = scan_settings['downsample_size']
    n_sigma = scan_settings['n_sigma']

    # inner loop step
    step_in = settings['step_in']

    # inner loop resample freq
    resample_freq = settings['resample_freq']

    # thresh to accept
    normality_thresh_in = settings['normality_thresh_in']

    # thresh to stop
    normality_thresh_out = settings['normality_thresh_out']

    # thresh to keep
    normality_thresh_keep = settings['normality_thresh_keep']

    # forward size if empty
    forward_size = settings['forward_size']*step_out

    # fallback size if after eating
    fallback_size = settings['fallback_size']*step_out

    print('Interpolating Dist_au...')

    # interpolate Dist_au with scipy.interpolate.interp1d
    ts = np.array([t.timestamp() for t in Dist_au.index])
    f_dist_au = interp1d(ts, Dist_au.values, 'linear', fill_value='extrapolate')

    # resample the Btot for inner loop
    Btot = Btot.resample(resample_freq).mean()

    # create Btot unix index
    Btot_index_unix = np.array([t.timestamp() for t in Btot.index])


    tstart = df_scans.index[0]
    tend = tstart + win
    scans = []
    streams = []
    while True:
            
        normality = df_scans.loc[tstart,'sw']
        nan_ratio = df_scans.loc[tstart,'nan_ratio']
        
        scans_in = []
            
        # found good interval!
        if normality >= normality_thresh_in:
            print("Found!: tstart=%s, tend=%s, normality=%.4f, len = %s" %(tstart, tend, normality, tend-tstart))

            # set tstart and tend in the inner loop
            tstart_in = tstart
            tend_in = tend + step_in
            
            # loop to eat the time series
            while True:
                # normality test
                ind = (Btot.index >= tstart_in) & (Btot.index < tend_in)
                indices = Btot.index[ind]
                btot = Btot[indices].values

                # get r from the interpolation
                ts = Btot_index_unix[ind]
                r = f_dist_au(ts)

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
                mean = np.mean(btot1)
                std = np.std(btot1)
                keep_ind = (btot1 > mean - n_sigma*std) & (btot1 < mean + n_sigma*std)
                btot1[np.invert(keep_ind)] = np.nan
                nan_ratio = np.sum(np.isnan(btot1))/len(btot1)
                x = btot1[np.invert(np.isnan(btot1))]

                normalities = {}    
                for method, ms in methods.items():
                    if len(x) < 500:
                        # if not enough samples, continue
                        normality = np.nan
                    else:
                        # normalize input for normality test
                        x = (x-np.mean(x))/np.std(x)

                        # normality test
                        if method == 'shapiro':
                            a1 = np.array([shapiro(np.random.choice(x, size=downsample_size, replace = False)).pvalue for i1 in range(ms['Ntests'])])
                        elif method == 'kstest':
                            # normalize the distribution for kstest
                            a1 = np.array([kstest(np.random.choice(x, size=downsample_size, replace = False), 'norm').pvalue for i1 in range(ms['Ntests'])])
                        else:
                            raise ValueError("Wrong method")
                        normality = np.sum(a1 > 0.05)/len(a1)
                    
                    normalities[method] = normality
                
                scan_in = {
                    't0': tstart_in,
                    't1': tend_in,
                    'nan_ratio': nan_ratio,
                    'normalities': normalities,
                    'r_ratio': r_ratio,
                    'settings': settings,
                    'fit_results': rfit,
                    'normality': normalities['shapiro']
                }
                
                # append the scan_in
                scans_in.append(scan_in)
                normality_in = scan_in['normality']
                nan_ratio_in = scan_in['nan_ratio']
                rfit_in = scan_in['fit_results']

                # create scans_in dataframe
                df = pd.DataFrame(scans_in)
                max_norm = df['normality'].max()

                # end the eating process if normality < thresh
                if normality_in < normality_thresh_out:
                    
                    # print the first, last, and best record
                    ii = np.argmax(df['normality'])
                    
                    print("")
                    print(" Thresh out: %.4f " %(normality_thresh_out))
                    print("first: tstart=%s, tend=%s, normality=%.4f, rfit=%.4f, len = %.2f Hr" %(
                        scans_in[0]['t0'], scans_in[0]['t1'], scans_in[0]['normality'], scans_in[0]['fit_results'], (scans_in[0]['t1']-scans_in[0]['t0'])/np.timedelta64(1, 'h')
                        ))
                    print("last: tstart=%s, tend=%s, normality=%.4f, rfit=%.4f, len = %.2f Hr" %(
                        scans_in[-1]['t0'], scans_in[-1]['t1'], scans_in[-1]['normality'], scans_in[-1]['fit_results'], (scans_in[-1]['t1']-scans_in[-1]['t0'])/np.timedelta64(1, 'h')
                        ))
                    ii = np.argmax(df['normality'])
                    print("best: tstart=%s, tend=%s, normality=%.4f, rfit=%.4f, len = %.2f Hr" %(
                        scans_in[ii]['t0'], scans_in[ii]['t1'], scans_in[ii]['normality'], scans_in[ii]['fit_results'], (scans_in[ii]['t1']-scans_in[ii]['t0'])/np.timedelta64(1, 'h')
                        ))
                    print("")
                    
                    
                    # keep the interval up until normality_thresh_keep
                    print(" Thresh keep: %.4f " %(normality_thresh_keep))
                    while(len(scans_in) > 0):
                        # pop the last one if with bad normality
                        if scans_in[-1]['normality'] < normality_thresh_keep:
                            pop_out = scans_in.pop()
                            print("poping: tstart=%s, tend=%s, normality=%.4f, rfit=%.4f, len = %.2f Hr" %(
                                pop_out['t0'], pop_out['t1'], pop_out['normality'], pop_out['fit_results'], (pop_out['t1']-pop_out['t0'])/np.timedelta64(1, 'h')
                                ))
                        # if the last one is good, break
                        else:
                            print("")
                            print("keeping: tstart=%s, tend=%s, normality=%.4f, rfit=%.4f, len = %.2f Hr, len = %d" %(
                                scans_in[-1]['t0'], scans_in[-1]['t1'], scans_in[-1]['normality'], scans_in[-1]['fit_results'], (scans_in[-1]['t1']-scans_in[-1]['t0'])/np.timedelta64(1, 'h'), len(scans_in)
                                ))
                            # set the new starting point
                            print("")
                            print("Finished Eating!! Falling back, resetting tstart = %s - %s" %(scans_in[-1]['t1'], fallback_size))
                            tstart = scans_in[-1]['t1'] - fallback_size
                            tend = tstart + win
                            # store the inner loop scanning records
                            streams.append(scans_in)
                            break
                            
                    
                    # if nothing left, go forward with forward_size
                    if len(scans_in) == 0:
                        print("")
                        print("Nothing left!! Fast-Forwarding, resetting tstart = %s + %s" %(tstart, forward_size))
                        tstart = tstart + forward_size
                        tend = tstart + win
                        
                    # if something left, tstart and tend has already been set
                    else:
                        pass
                        
                    
                    # end the inner loop
                    print("")
                    break
                # otherwise eat one more step
                else:
                    print("Eating...: tstart=%s, tend=%s, normality=%.4f, nan_ratio = %.4f, rfit = %.4f, len = %s" %(tstart_in, tend_in, normality_in, nan_ratio_in, rfit_in, tend_in - tstart_in))
                    tend_in = tend_in + step_in    
                    
                # end the loop if it reach the end
                if tend_in > Btot.index[-1]:
                    tend = tend_in
                    # store the inner loop scanning records
                    streams.append(scans_in)
                    break
                
        # move to next window
        else:
            tspent = (time.time() - tl0)
            ratio = (tend-Btot.index[0])/(Btot.index[-1]-Btot.index[0])
            # clear_output(wait=True)
            print("scanning...: tstart=%s, tend=%s, normality=%.4f, nan_ratio=%.4f, finished = %.2f%%, time spent = %.2f min, time left = %.2f min" %(
                tstart, 
                tend, 
                normality, 
                nan_ratio,
                (tend-Btot.index[0])/(Btot.index[-1]-Btot.index[0])*100,
                tspent/60,
                (tspent/ratio-tspent)/60
                ), end='\r')
            tstart = tstart + step_out
            tend = tend + step_out
        
        if tstart > df_scans.index[-1]:
            break  

    results = {
        'scans': scans,
        'streams': streams,
        'settings': settings
    }

    return results


def SolarWindScannerInnerLoopSequential(input):
    # allocate variables
    btot = input['btot']
    r = input['r']
    tstart = input['tstart']
    tend = input['tend']

    # normalize btot with r
    btot1 = btot * ((r/r[0])**2)
    
    # discard points outside of 3 sigma
    # solar wind has a much higher chance than gaussian to have extreme values
    mean = np.mean(btot1)
    std = np.std(btot1)
    keep_ind = (btot1 > mean - 3*std) & (btot1 < mean + 3*std)
    btot1[np.invert(keep_ind)] = np.nan
    x = btot1.values[np.invert(np.isnan(btot1))]

    if len(x) < 500:
        # if not enough samples, continue
        normality = np.nan
    else:
        # normality test
        a1 = np.array([shapiro(np.random.choice(x, size=500)).pvalue for i1 in range(10000)])
        normality = np.sum(a1 > 0.05)/len(a1)
    
    scan = {
        't0': tstart,
        't1': tend,
        'normality': normality
    }

    return scan


# some testing function for parallelization

# def testing(x, parallel = False):

#     if not parallel:
#         a1 = np.array([shapiro(np.random.choice(x, size=500)).pvalue for i1 in range(1000000)])
#     else:
#         with Pool(4) as p:
#             a1 = p.map(f, [x for i1 in range(1000000)])
#             a1 = np.array(a1)

#     return a1

# def f(x):
#     return shapiro(np.random.choice(x, size=500)).pvalue