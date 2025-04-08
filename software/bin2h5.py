import numpy as np
import scipy as sp
import os
import h5py
import matplotlib.pyplot as plt
from scipy import signal


labels = ['z-wave', 'wifi', 'lte', 'lora', 'zigbee']

#bw of radio used to collect data, this is fixed
bw_mhz = 25
#input dim to the neural network
buf = 1024
#bw resolution of a single IQ
resolution = bw_mhz/buf

#bw in mhz of each protocol
signal_bw_mhz_dict = {
    'wifi'      : 20,
    'lte'       : 10,
    'zigbee'    : 2,
    'lora'      : 0.5,
    'ble'       : 1,
    'z-wave'    : 1.5,
    #'unknown'   : 25     # change for unknown
}

#bw in number of iqs
signal_bw_mhz_niqs = {}
for label in signal_bw_mhz_dict:
    niqs = np.ceil(signal_bw_mhz_dict[label]/resolution)

    if niqs%2 == 1:
        niqs += 1

    signal_bw_mhz_niqs[label] = int(niqs)


#raw data folder
raw_folder_fp = '/media/gaofeng/Extreme SSD/datasets/spectrum_sensing_stitching/raw/'
raw_dir = os.listdir(raw_folder_fp)

#folder of signal bank where data will be put into
data_fp = '/media/gaofeng/Extreme SSD/datasets/spectrum_sensing_stitching/signal_bank/'
if not os.path.isdir(data_fp):  
    os.mkdir(data_fp)






#this value needs to be changed depending on the protocol used and the signal strength you deem relevant
#the values we used are found below, feel free to play with this
threshold = {
    'wifi': 30,
    'lte': 0.5,
    'zigbee': 0.13,
    'lora': 5,
    'ble': 0.01,
    'z-wave': 0.025,
    #'unknown': 0.02    # change for unknown 
}

#if you want to plot pre-processed data and data that will be discarded/not included (to test threshold choice)
plot = True
for label in labels:

    #contains signals of interest
    contains_signal = []
    #contains empty channel
    contains_empty = []
    energy_list = []


    #used for plotting: contains time domain version of signals of interest
    contains_signal_time = []
    #used for plotting: contains time domain version of irrelevant signals
    contains_irrelevant_signal_time = []

    if label != 'empty':
        for file in raw_dir:

            if os.path.isfile(os.path.join(raw_folder_fp,file)) and label in file and 'empty' not in file:

                with open(os.path.join(raw_folder_fp,file)) as binfile:
                    print(os.path.join(raw_folder_fp,file))

                    #set this value to less than buf if you want samples to overlap in time
                    stride = buf

                    #Extract IQs and group them based on input size
                    all_samps = np.fromfile(binfile, dtype=sp.complex64, count=-1, offset=0)
                    all_samps = np.array([all_samps[k:k + buf] for k in range(0, len(all_samps) - 1 - buf, stride)])

                    #mask that will be used to only keep band of interest
                    mask = np.ones(signal_bw_mhz_niqs[label], dtype=int)
                    mask_empty = np.zeros(buf-signal_bw_mhz_niqs[label], dtype=int)
                    final_mask = np.insert(mask_empty, len(mask_empty)//2, mask)


                    for samp_idx, samps in enumerate(all_samps):
                        #convert to freq domain and only grab bands of interest
                        sample_f = np.fft.fft(samps)
                        sample_f = np.fft.fftshift(sample_f)
                        sample_f[np.where(final_mask == 0)[0]] = 0

                        #Calculate energy and compare to threshold
                        power = np.abs(sample_f)**2
                        energy = np.sum(power)/len(power)

                        energy_list.append(energy)
                        if energy > threshold[label]:
                            contains_signal.append(np.transpose(np.stack((np.real(sample_f), np.imag(sample_f)))))
                            if plot:
                                contains_signal_time.append(np.transpose(np.stack((np.real(samps), np.imag(samps)))))
                        else:
                            if plot:
                                contains_irrelevant_signal_time.append(np.transpose(np.stack((np.real(samps), np.imag(samps)))))
                    print(max(energy_list), min(energy_list))
                    print("average energy: " + str(np.mean(energy_list)))
                    print("median energy: " + str(np.median(energy_list)))
                    print("lenth of power list: " + str(len(power)))
                    #print("frist 17 percent of energy list:", np.mean(energy_list[:int(len(energy_list) * 0.17)]))
                    print("number of samples: " + str(len(energy_list)))


        contains_signal = np.array(contains_signal)

        print("Number of Signal Samples: " + str(contains_signal.shape[0]))

        '''signal_filename = data_fp + label + '.bin'
        contains_signal.tofile(signal_filename)'''

        #store in h5 file
        '''f_signal = h5py.File(data_fp + label + '.h5', 'w')
        dset = f_signal.create_dataset(label, (contains_signal.shape[0], contains_signal.shape[1], contains_signal.shape[2]), dtype='f')
        dset[()] = contains_signal
        f_signal.close()'''

        if plot:
            #This converts the contains_signal_time list into a NumPy array. Each element in contains_signal_time is a time-domain representation of the signal.
            contains_signal_time = np.array(contains_signal_time)
            contains_irrelevant_signal_time = np.array(contains_irrelevant_signal_time)

            #This reconstructs the complex I/Q signal from its real and imaginary parts
            contains_signal_iq = contains_signal_time[:,:,0] + 1j* contains_signal_time[:,:,1]

            #This flattens the multi-dimensional contains_signal_iq array into a 1D array and extracts the first 10% of its samples.
            #This computes the spectrogram of the signal. A spectrogram is a visual representation of how the signal's power spectral density (PSD) changes over time.
            f_signal, t_signal, Sxx_signal = signal.spectrogram(contains_signal_iq.flatten()[:int(len(contains_signal_iq.flatten()) * 0.1)], 25_000_000, return_onesided=False, nperseg=buf, noverlap=0)

            #The output of the spectrogram has frequency components arranged from 0 to the Nyquist frequency (positive frequencies) and then from the Nyquist frequency to negative frequencies.
            #fftshift reorders these frequencies to place 0 Hz in the middle, making the spectrogram symmetric around the center frequency.
            Sxx_signal = np.fft.fftshift(Sxx_signal, axes=0)

            plt.pcolormesh(t_signal, np.fft.fftshift(f_signal), Sxx_signal, shading='auto', vmax=np.max(Sxx_signal)/100)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('Signal ' + label)
            plt.show()

            contains_irrelevant_iq = contains_irrelevant_signal_time[:,:,0] + 1j* contains_irrelevant_signal_time[:,:,1]

            f_empty, t_empty, Sxx_empty = signal.spectrogram(contains_irrelevant_iq.flatten()[:int(len(contains_irrelevant_iq.flatten()) * 0.1)], 25_000_000, return_onesided=False, nperseg=buf, noverlap=0)
            Sxx_empty = np.fft.fftshift(Sxx_empty, axes=0)
            plt.pcolormesh(t_empty, np.fft.fftshift(f_empty), Sxx_empty, shading='auto', vmax=np.max(Sxx_signal)/100)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('Empty ' + label)
            plt.show()
        print('end')

    else:

        for file in raw_dir:
            if os.path.isfile(os.path.join(raw_folder_fp, file)) and label in file and 'empty' in file:

                with open(os.path.join(raw_folder_fp, file)) as binfile:
                    print(os.path.join(raw_folder_fp, file))

                    #set this value to less than buf if you want samples to overlap in time
                    stride = buf

                    #extract iqs
                    all_samps = np.fromfile(binfile, dtype=sp.complex64, count=-1, offset=0)
                    all_samps = np.array([all_samps[k:k + buf] for k in range(0, len(all_samps) - 1 - buf, stride)])
                    for samp_idx, samps in enumerate(all_samps):
                        #convert to freq and store
                        sample_f = np.fft.fft(samps)
                        sample_f = np.fft.fftshift(sample_f)

                        contains_empty.append(np.transpose(np.stack((np.real(sample_f), np.imag(sample_f)))))

        contains_empty = np.array(contains_empty)

        print("Number of Empty Samples: " + str(contains_empty.shape[0]))

        f_signal = h5py.File(data_fp + label + '.h5', 'w')
        dset = f_signal.create_dataset(label, (contains_empty.shape[0], contains_empty.shape[1], contains_empty.shape[2]), dtype='f')
        dset[()] = contains_empty
        f_signal.close()