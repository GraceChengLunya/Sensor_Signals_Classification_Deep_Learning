import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

###################################
### PROJECT SPECIFIC PARAMETERS ###
###################################

#if enabled will plot a sample
debug = False
#total number of training/testing samples to generate
nsamples = 600_000
#max number of signals to be plotted at the same time in frequency
max_simultaneous_signals = 2
#total observable bandwidth (for our signal bank this needs to be a multiple of 25MHz)
bw = 25_000_000
#input size to the NN (for our signal bank this needs to be a multiple of 1024)
buf =1024
#frequency resolution or bw of a single bin in the NN input
resolution = (bw/1000000)/buf
#probabliity that the entire observable bandwith doesnt contain any signal
prob_empty = 0.05
#probability that the first signal injected into the input is centered
prob_centered = 0.5

nclasses = 6  # change for unknown

#dict containing indexes of rows belonging to each protocol in classification label matrix
#(note "empty" doesnt get its own row in the matrix) you can think of these as class labels
#where each value corresponds to the row that will contain 1 in columns/IQs where that signal is present
#and 0 where it isnt
label_dict_str= {
    'wifi'      : 0,
    'lte'       : 1,
    'zigbee'    : 2,
    'lora'      : 3,
    'ble'       : 4,
    'z-wave'    : 5
}

#flipped version of above dict
label_dict_int = dict([(value, key) for key, value in label_dict_str.items()])

#dict of bw for each signal in mhz
signal_bw_mhz_dict = {
    'wifi'      : 20,
    'lte'       : 10,
    'zigbee'    : 2,
    'lora'      : 0.5,
    'ble'       : 1,
    'z-wave'    : 1.5
}

#dict of bw for each signal in number of IQs
signal_bw_mhz_niqs = {}
for label in signal_bw_mhz_dict:
    niqs = np.ceil(signal_bw_mhz_dict[label]/resolution)

    if niqs%2 == 1:
        niqs += 1

    signal_bw_mhz_niqs[label] = int(niqs)

#path to signal bank dir
#each class should have its own h5 file i.e. lora.h5, wifi.h5 etc
data_fp = '/media/gaofeng/Extreme SSD/datasets/spectrum_sensing_stitching/signal_bank/'
final_folder_fp = '/media/gaofeng/Extreme SSD/datasets/spectrum_sensing_stitching/processed/'
if not os.path.isdir(final_folder_fp):
    os.mkdir(final_folder_fp)

###################################
### PROJECT SPECIFIC PARAMETERS ###
###################################


#grab random signal from signal bank
def get_sample(protocol):
    f_signal = h5py.File(data_fp + protocol + '.h5', 'r')
    samp = f_signal[protocol][np.random.randint(f_signal[protocol].shape[0])]
    f_signal.close()
    return samp


if __name__ == '__main__':

    all_labels = []
    all_inputs = []

    #if you want to omit a class you can remove them from below list
    protocols_used = [0,1,2,3,4,5]                                                        # change for unknown
    for i in tqdm(range(nsamples)):
        label = np.zeros([nclasses,buf], dtype=int)

        input_samp = np.zeros([buf, 2])

        if np.random.rand() > prob_empty:
            nsignals = np.random.randint(1,max_simultaneous_signals+1)

            signal_protocols = np.zeros(nsignals)
            signal_freqs = np.zeros(nsignals)
            prev_centered = False

            for j in range(nsignals):
                protocol = np.random.choice(protocols_used)

                signal_protocols[j] = protocol

                signal_bw = signal_bw_mhz_niqs[label_dict_int[protocol]]
                temp_input = np.zeros([buf+signal_bw*2-2,2])

                samp = get_sample(label_dict_int[protocol])
                samp_cut = samp[buf//2-signal_bw//2:buf//2+signal_bw//2,:]

                if np.random.rand() < prob_centered and not prev_centered:
                    signal_location = int(len(temp_input)/2-signal_bw/2)
                    prev_centered = True
                else:
                    signal_location = int(np.random.choice(len(temp_input)-signal_bw+1))



                temp_mask = np.zeros(buf+signal_bw*2-2, dtype=bool)
                temp_mask[signal_location:signal_location + signal_bw] = True

                signal_freqs[j] = signal_location*resolution

                temp_input[signal_location:signal_location+signal_bw,:] += samp_cut
                label[protocol] |= temp_mask[signal_bw-1:signal_bw+buf-1]
                input_samp += temp_input[signal_bw-1:signal_bw+buf-1,:]


        input_samp += get_sample('empty')
        all_inputs.append(input_samp)
        all_labels.append(label)


        if debug:
            plt.plot(np.linspace(-bw/1_000_000/2,bw/1_000_000/2,buf),abs(input_samp[:,0] + 1j*input_samp[:,1]))
            plt.xlabel('MHz')
            plt.title(signal_protocols)
            plt.show()

    all_inputs = np.array(all_inputs)
    all_labels = np.array(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        all_inputs, all_labels, test_size=0.1, random_state=42)

    f_test = h5py.File(
        final_folder_fp + 'test.h5', 'w')
    xtest = f_test.create_dataset('X', (X_test.shape[0], X_test.shape[1], X_test.shape[2]), dtype='f')
    ytest = f_test.create_dataset('y', (y_test.shape[0],y_test.shape[1], y_test.shape[2]), dtype='i')

    xtest[()] = X_test
    ytest[()] = y_test

    f_train = h5py.File(
        final_folder_fp + 'train.h5', 'w')
    xtrain = f_train.create_dataset('X', (X_train.shape[0], X_train.shape[1], X_train.shape[2]), dtype='f')
    ytrain = f_train.create_dataset('y', (y_train.shape[0],y_train.shape[1], y_train.shape[2]), dtype='i')

    xtrain[()] = X_train
    ytrain[()] = y_train

    f_test.close()
    f_train.close()
