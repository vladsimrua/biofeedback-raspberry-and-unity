#import matplotlib.pyplot as plt
#from scipy import signal
#import pygame

#import max30102
import time
import socket
import numpy as np
import math
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features, plot_psd, get_geometrical_features, get_frequency_domain_features
#from threading import Thread

BUFFER_SIZE = 200
SAMPLE_FREQ = 25
# taking moving average of 4 samples when calculating HR
# in algorithm.h, "DONOT CHANGE" comment is attached
MA_SIZE = 4
TIME_SIMPLE_FROZE = 38  # 38 ms = 0.038 s


# [836, 798, 798, 532, 266] - 3a

# [836, 836, 798, 494, 342] - 4a
# [836, 836, 798, 494, 342][494, 266, 684, 722, 684][798, 874, 570, 266, 912][798, 798, 760, 760]


def peaks(ir):
    #ir.reverse()
    # plt.plot(list2)
    # plt.show()

    ir_mean = int(np.mean(ir))

    x = -1 * (np.array(ir) - ir_mean)
    for i in range(x.shape[0] - MA_SIZE):
        x[i] = np.sum(x[i:i + MA_SIZE]) / MA_SIZE

    n_th = int(np.mean(x))
    n_th = 30 if n_th < 30 else n_th  # min allowed
    n_th = 60 if n_th > 60 else n_th  # max allowed

    ir_valley_locs, n_peaks = find_peaks(x, BUFFER_SIZE, n_th, 4, 15)
    peak_rr = []
    peak_interval_sum = 0
    if n_peaks >= 2:
        for i in range(1, n_peaks):
            peak_interval_sum += (ir_valley_locs[i] - ir_valley_locs[i - 1])
            peak_rr.append(TIME_SIMPLE_FROZE * (ir_valley_locs[i] - ir_valley_locs[i - 1]))
        peak_interval_sum = int(peak_interval_sum / (n_peaks - 1))
    # print(peak_rr)
    return peak_rr

# this assumes ir_data and red_data as np.array

def find_peaks(x, size, min_height, min_dist, max_num):
    """
    Find at most MAX_NUM peaks above MIN_HEIGHT separated by at least MIN_DISTANCE
    """
    ir_valley_locs, n_peaks = find_peaks_above_min_height(x, size, min_height, max_num)
    ir_valley_locs, n_peaks = remove_close_peaks(n_peaks, ir_valley_locs, x, min_dist)

    n_peaks = min([n_peaks, max_num])

    return ir_valley_locs, n_peaks


def find_peaks_above_min_height(x, size, min_height, max_num):
    """
    Find all peaks above MIN_HEIGHT
    """

    i = 0
    n_peaks = 0
    ir_valley_locs = []  # [0 for i in range(max_num)]
    while i < size - 1:
        if x[i] > min_height and x[i] > x[i - 1]:  # find the left edge of potential peaks
            n_width = 1
            # original condition i+n_width < size may cause IndexError
            # so I changed the condition to i+n_width < size - 1
            while i + n_width < size - 1 and x[i] == x[i + n_width]:  # find flat peaks
                n_width += 1
            if x[i] > x[i + n_width] and n_peaks < max_num:  # find the right edge of peaks
                # ir_valley_locs[n_peaks] = i
                ir_valley_locs.append(i)
                n_peaks += 1  # original uses post increment
                i += n_width + 1
            else:
                i += n_width
        else:
            i += 1

    return ir_valley_locs, n_peaks


def remove_close_peaks(n_peaks, ir_valley_locs, x, min_dist):
    """
    Remove peaks separated by less than MIN_DISTANCE
    """

    # should be equal to maxim_sort_indices_descend
    # order peaks from large to small
    # should ignore index:0
    sorted_indices = sorted(ir_valley_locs, key=lambda i: x[i])
    sorted_indices.reverse()

    # this "for" loop expression does not check finish condition
    # for i in range(-1, n_peaks):
    i = -1
    while i < n_peaks:
        old_n_peaks = n_peaks
        n_peaks = i + 1
        # this "for" loop expression does not check finish condition
        # for j in (i + 1, old_n_peaks):
        j = i + 1
        while j < old_n_peaks:
            n_dist = (sorted_indices[j] - sorted_indices[i]) if i != -1 else (
                    sorted_indices[j] + 1)  # lag-zero peak of autocorr is at index -1
            if n_dist > min_dist or n_dist < -1 * min_dist:
                sorted_indices[n_peaks] = sorted_indices[j]
                n_peaks += 1  # original uses post increment
            j += 1
        i += 1

    sorted_indices[:n_peaks] = sorted(sorted_indices[:n_peaks])

    return sorted_indices, n_peaks

def obrabotka(rr_int):
    #print("start obrabotka")
    rr_intervals_list = rr_int
    flag = True

    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals_list,
                                                    low_rri=500, high_rri=2000)
    #rint("R-R intervals without_outliers: ",rr_intervals_without_outliers)

    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                       interpolation_method="linear")
    #print("R-R intervals interpolated: ", rr_intervals_without_outliers)
    interpolated_rr_intervals_not_nan = []
    n_nan = 0
    for ad in range(len(interpolated_rr_intervals)):
        if math.isnan(float(interpolated_rr_intervals[ad])):
            n_nan = n_nan + 1
        else:
            interpolated_rr_intervals_not_nan.append(interpolated_rr_intervals[ad])
    # This remove ectopic beats from signal
    #print("R-R intervals interpolated not nun: ",interpolated_rr_intervals_not_nan)
    if len(interpolated_rr_intervals_not_nan) == 0:
        flag = False

    if len(interpolated_rr_intervals_not_nan) == 1:
        flag = False

    if flag == False:
        return flag, interpolated_rr_intervals_not_nan
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals_not_nan, method="malik")
    #print("N-N intervals: ",nn_intervals_list)

    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    #print("N-N intervals interpolated: ", interpolated_nn_intervals)

    # time_domain_features = get_time_domain_features(nn_intervals_list)
    interpolated_nn_intervals_not_nan= []
    n_nan= 0
    for ad in range(len(interpolated_nn_intervals)):
        if math.isnan(float(interpolated_nn_intervals[ad])):
           n_nan=n_nan+1
        else:
            interpolated_nn_intervals_not_nan.append(interpolated_nn_intervals[ad])
    #interpolated_nn_intervals_not_nan = interpolated_nn_intervals
    #print("n nan: ", n_nan)
    print("N-N intervals interpolated not nan: ", interpolated_nn_intervals_not_nan)

    if len(interpolated_nn_intervals_not_nan)==0:
        flag = False

    if len(interpolated_nn_intervals_not_nan)==1:
        flag = False
  #  time_domain_features = get_time_domain_features(interpolated_nn_intervals_not_nan)
    #print("end obrabotka")
    return flag, interpolated_nn_intervals_not_nan
    #d = time.time() - t
    # plt.plot(interpolated_nn_intervals)
    # plt.show()
    # print("computation time: ",d)
    #print("new hr: ", time_domain_features.get("mean_hr"))


if __name__ == '__main__':
    # rr = print_hi("3")
    # print(rr)
    # print(a) 71,77 75,187 121,45 = 68 (85)
    rr_intervals_list_all = []
    nn_intervals_list_all = []
   # rr_intervals_list = [494, 266, 684, 722, 684, 798, 874, 570, 266, 912, 798, 798, 760, 760]

    hr_list =[]
    rr_mean_list=[]
    rr_median_list=[]

    sdnn_list=[]
    rmssd_list= []
    nni_50_list=[]
    pnni_50_list=[]
    epoch = 1
    rr_intervals_list = []
    epoch= 0
    m = max30102.MAX30102()
    host, port = "127.0.0.1", 25001
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    while True:
        #time.sleep(0.5)  # sleep 0.5 sec
        red, ir = m.read_sequential()
        print("i=", epoch, "ir = ", ir)
        rr_intervals_list = peaks(ir)
        print("rr interval: ", rr_intervals_list)
        flag, interpolated_nn_intervals_not_nan = obrabotka(rr_intervals_list)
        if flag == True:

            time_domain_features = get_time_domain_features(interpolated_nn_intervals_not_nan)
            plot_psd(interpolated_nn_intervals_not_nan, method="lomb")
            for i in range(len(interpolated_nn_intervals_not_nan)):
                t = int(interpolated_nn_intervals_not_nan[i])
                nn_intervals_list_all.append(t)
                hr, hr_v, spo, spo_v = hrcalc.calc_hr_and_spo2(ir, red))
                mean_hr = int(time_domain_features.key("mean_hr"))
                mean_rr = int(time_domain_features.key("mean_nni"))
                median_rr = int(time_domain_features.key("median_nni"))

                sdnn = int(time_domain_features.key("sndd"))
                rmssd = int(time_domain_features.key("rmssd"))
                spo2 = int(spo)
                if math.isnan(mean_hr) and math.isnan(mean_rr) and math.isnan(median_rr) and math.isnan(sdnn) and math.isnan(rmssd) and math.isnan(spo2):
                    sock.sendall(str(mean_hr).encode("UTF-8"))  # Converting string to Byte, and sending it to C#
                    sock.sendall(str(spo).encode("UTF-8"))
                    sock.sendall(str(mean_rr).encode("UTF-8"))
                    sock.sendall(str(median_rr).encode("UTF-8"))
                    sock.sendall(str(rmssd).encode("UTF-8"))
                    sock.sendall(str(sdnn).encode("UTF-8"))

                    receivedData = sock.recv(1024).decode(
                        "UTF-8")  # receiveing data in Byte fron C#, and converting it to String
                    print(receivedData)
