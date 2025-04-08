# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Our project uses USRP N210 Hardware Drives (UHD) and GNU Radio to collect wireless signals from smart home sensors of different communication protocols such as WiFi, Zigbee, Z-Wave, etc. We use a spectrum sensing model called Stitching-the-Spectrum to process signal data. We would also use a U-Net based CNN Deep Learning model to classify the signal type and eventually identify those sensors without prior knowledge of their types. We plan to develop a classification model that autonomously and accurately discovers and classifies wireless sensors in a smart home by transmitted wireless signals.

# 1. Introduction

* Motivation & Objective:
  
  Many smart homes are equipped with sensors that collect wireless data about the environment and how people interact with their spaces. Yet, much of this information isn’t fully utilized to understand daily activities and routines. By combining this sensor data with common knowledge about how devices like door, motion, and multi sensors are typically used, we can gain deeper insight into occupants’ behaviors. Our objective is to develop a system that uses advanced AI models to interpret these wireless signals. This will enhance the smart home experience, making it more responsive and intuitive to the needs of its users.
  
* State of the Art & Its Limitations:
  
  You Only Look Once (YOLO) is a current widely used object detection model that divides the input (usually an image) into grids. For signal classification, YOLO divide a signal into fixed-length windows, or bounding boxes.

Limitations of You Only Look Once (YOLO):

This approach requires the creation of an image out of in-phase/quadrature (I/Q) samples, incurring additional latency.
Being a viable method for computer vision tasks, YOLO does not achieve the level of resolution required for wireless signals.
Modern wireless signals such as 5G and LoRa can hardly confirm into square bounding boxes which leads to a significant amount of spectrum being incorrectly classified as occupied, leading to poor spectrum efficiency.

* Novelty & Rationale: What is new in your approach and why do you think it will be successful?
  The approach we found in a paper is called Stitching-the-Spectrum, which “stitches” different signals together to create samples where signals are overlapping and affected by real-world noises and inferences.

Novel dataset generation pipeline that generates large-scale datasets that

contains signals collected Over-the-Air (OTA) and are affected by real-world conditions
can be completely labeled
low latency
Novel custom DL algorithm for multi-label multi-class spectrum sensing based on semantic segmentation that

operates at the I/Q level instead of creating images
classifies each I/Q sample incoming from the ADC without creating bounding boxes, thus increasing classification accuracy significantly
only uses 1024 I/Q samples as input, which leads to very low inference time.

* Potential Impact:
  Technical impacts:

The network is able to achieve more accuracy (7% in the paper) than U-Net in the most challenging protocols (Wi-Fi and LTE) while maintaining similarly low latency
The dataset generation pipeline makes the wideband spectrum sensing algorithm more accurate even with data collected from different devices, sampling rates, antennas.
Intersection over Union (IoU) of 96.70%; 2.6ms of latency to process 100MHz of spectrum.
Broad impacts:

Less amount of data needed for understanding the signal (protocol, MAC address, etc.)
A more accurate classification result that takes less time
Significantly lower the difficulty of learning about the information of sensors (their types and locations) in a smart home without prior knowledge

* Challenges:
  
  Data must be collected Over-the-Air (OTA) to capture real-world channel conditions and prepare the AI in a wireless setting with real radios
OTA data involves many possible combinations of time an frequency, which significantly increases the complexity in the case of wideband spectrum sensing
OTA data requires large portions of spectrum without any interference from external systems
Training data that assumes synchronization between sensing and transmission center frequencies results in poor accuracy

* Requirements for Success:
  
Resources:

SDRs (USRP N210)
Smart sensors that use different protocols
GPUs

Skills:

The usage of USRP N210 Software Defined Radio (SDR) for collecting wireless signals
Signal Processing skills with GNU Radio or languages like python and MATLAB, etc.
Understanding of Deep Learning and RF signals

* Metrics of Success:
  
 For protocol classification:

Intersection of Union (IoU): Formula: Area of Intersection / Area of Union to indicate the quality of math between the predicted and ground truth boxes. (1 means perfect overlap and 0 means no overlap at all)

# 2. Related Work

YOLO (You Only Look Once)

YOLO is a real-time object detection model that treat object detection as regression problems. The YOLO networks process the entire image at once, predicting bounding boxes and class probabilities at the same time. Compared to traditional two-stage detectors that first generate regional proposals then classify them, YOLO directly outputs class predictions and location coordinates to achieve near real-time performance. Later versions like v2, v3 and v4 aim to improve accuracy and lower latency. 

In conclusion, YOLO detects all objects within an image

![image](https://github.com/user-attachments/assets/8662e6d7-3e25-4448-939c-18c449e480eb)

U-Net

U-Net is a convolutional neural network architecture primarily designed for precise and efficient image segmentation tasks. Developped for biomedical image segmentation. This architecture uses skip connections connecting downsampling and upsampling layers to retain spatial information lost during pooling 

In conclusion, U-Net achieves fine segmentation results even with small training datasets, which makes it popular in medical imaging.

# 3. Technical Approach

* Data Collection
  
  To train our deep learning classification model, we use both of our collected data from the lab (Wi-Fi and z-wave) and other signal types from the Stitching-the-Spectrum model. To collect our own data, we use USRP N210 to collect signals with a bandwith of 25MHz generated by our sensors and turn them into binary files (.bin) using GNU Radio.

![image](https://github.com/user-attachments/assets/4c0eb48f-ef75-4194-904f-2b63726cfee0)
![image](https://github.com/user-attachments/assets/886423c1-e887-4b2b-af9a-692b8ff26676)
![image](https://github.com/user-attachments/assets/e1d98ed0-70f3-40f5-a22c-70f34f14fe00)

* Training Data Processing
  
  Once we have collected signal data for each signal type, we would pre-process the IQ data by removing silence period and extracting signal of interest. First, we apply a bandpass filter using a frequency mask to extract relevant parts of the spectrum and remove any undesired signals that may have been recorded. Once pre-processed, the signals are converted to the frequency domain through a Fast Fourier Transform (FFT). Lastly, in the frequency domain, frequency components outside of the band of interest occupied by the signal are pruned, and the remaining data would be added to the signal bank. Threshold is also introduced to each signal type to remove signal of irrelevant strength (noise). For each data type, this procedure is repeated multiple times to generate a final signal bank[3].

  The signal bank consists of a labeled collection of signals collected when only one signal is transmitted at a time, with known bandwidth and center frequency. The data collection also made sure that the cleanest signal possible is collected by performing the collection over a limited and small portion of the spectrum without interference from other signals[3].

  <img width="487" alt="Screenshot 2024-12-11 at 1 12 16 PM" src="https://github.com/user-attachments/assets/91374aca-0090-4386-a894-201f1d2536e2" />

  <img width="481" alt="Screenshot 2024-12-11 at 12 59 49 PM" src="https://github.com/user-attachments/assets/923565d7-bd6a-4c3c-8d36-70fe7a1f120c" />

* Dataset Generation
  
  With signal banks corresponding to each signal type, the semi-augmented dataset generator pipeline would combine those signal banks to generate a "stitched" wideband signal to be added to the training dataset[3]. Relevant parameters include the total number of signal types C, the desired observable bandwidth B, maximum number of signals n_s which can be present in B at a given time, the probablity p_e that the entire observed bandwidth is empty, and the probablity p_c that any one of the signals is located at the center frequency, etc.

  With all these parameters, the pipeline generates M <n_s of signals to be injected into the bandwidth using the following formula.


<img width="175" alt="Screenshot 2024-12-13 at 11 53 20 PM" src="https://github.com/user-attachments/assets/1f8aed2b-b9d7-4559-bda5-8334e41d0bdf" />


After M signals are generated, the pipeline assigns a target class to each of the signal. The next step would be determining the position (center frequency) of the signal using


<img width="261" alt="Screenshot 2024-12-13 at 11 55 19 PM" src="https://github.com/user-attachments/assets/7e865517-1811-4a47-9bd7-7f300741dc2e" />


Eventually, labels are structured as a matrix L of C x n_iq, where n_iq is the number of I/Q's fed to the Deep Learning model.

* Semantic Spectrum Segmentation
  
  Our U-Net based DL model can classify I/Q samples being fed to it. The architecture is inspired by U-Net but with 1D CNN kernels instead of 2D. There are totally 5 encoding and decoding block.

  The encoding block has two 1D convolutional layers followed by batch normalization and ReLu activation. There is also a maxpooling layer downsampling the encoded features. The decoding block would take both output features from the previous layer and the encoded features from a skip connection. Last but not least, a non-locol block is introduced to mix up features globally to improve the performance of the system[3].

  <img width="348" alt="Screenshot 2024-12-14 at 12 10 10 AM" src="https://github.com/user-attachments/assets/02b72d4b-df01-444a-bff6-db9915b5c63d" />


# 4. Evaluation and Results

* Evaluation Metrics - Intersection over Union (IoU):
  
  IoU was calculated to measure the overlap between the model’s predictions and the ground truth. Higher IoU indicates better alignment with the true signal segmentation.
  ![image](https://github.com/user-attachments/assets/11582a30-a7d8-4259-951d-5f4649b9547d)
  ![image](https://github.com/user-attachments/assets/df6de275-a504-4152-9808-1ad661452497)

* The results demonstrated that the model performed well for most protocols and it achieved high IoU scores for BLE (0.9507), LoRa (0.9940), Zigbee (0.8877), and Z-Wave (0.9802). However, the performance for WiFi (0.3969) and LTE (0.6954) was significantly lower. There are some potential reasons. First, the thresholding used during data preprocessing might not have been optimal for WiFi and LTE signals, leading to inaccurate segmentation. Second, the overlap between WiFi and LTE in the spectrum could have introduced ambiguity, which makes it difficult for the model to distinguish between two protocols. Lastly, the WiFi signals in the dataset may not have been clean enough. Chances are the noise or interference degraded performance. Addressing these issues through better thresholding, improved data quality, and enhanced preprocessing could help improve the model's accuracy for these protocols.


![image](https://github.com/user-attachments/assets/73baee3d-61bf-4450-a1cd-8e35bd8c2991)
* This is the evaluation results for Z-Wave. It demonstrates that the model successfully identifies the Z-Wave protocol with high accuracy, as shown in the spectrogram inference. However, there are instances of misclassification between Z-Wave signals, where noise or unrelated signals are incorrectly labeled as LTE. A potential reason for this misclassification is the absence of an "unknown" class in the model. Without this class, the model is forced to assign a label to all input signals, leading to noise being classified as LTE. 


![image](https://github.com/user-attachments/assets/74d41ed2-a84c-49d4-adc5-67af6bcbc8d4)
* The is the evaluation results for WiFi. Some portions of WiFi signals are identified as LTE. One potential reason for this issue is that there are indeed overlapping LTE signals in the recorded data. Another reason might be true misclassification resulting from our model'a accuracy limitations, particularly in distinguishing between WiFi and LTE due to their overlapping spectral characteristics or insufficient feature differentiation. Since we use recorded data from nesl lab, where the environment is more noisy, the presence of overlapping LTE signals in our recorded data might confuse the model.


![image](https://github.com/user-attachments/assets/307d6e21-8805-40cd-9c4f-e7fb99075ab9)
* The evaluation results for Zigbee shows that the model successfully identifies the Zigbee protocol with high accuracy and it indicated by the distinct identification of the Zigbee signals in the spectrogram inference. However, there are some instances of misclassified signals, where noise or unrelated signals are incorrectly labeled as WiFi. A potential reason for this misclassification is the absence of an "unknown" class in the model, which forces it to assign a known label to all signals. This leads to noise being misclassified as WiFi. 


![image](https://github.com/user-attachments/assets/174c1c1d-3333-492e-b1ea-44595dac07db)
* The evaluation results for LTE shows that the model performs well in identifying LTE signals, as evidenced by the high accuracy in the inference plot. The LTE segments are correctly classified, with minimal misclassifications or noise interference observed. 


![image](https://github.com/user-attachments/assets/490ca7bf-199e-44a0-b471-0dbf7ba32b5b)
* This is the evaluation results for BLE. It demonstrates that the model successfully identifies the BLE protocol in many instances, as reflected in the inference plot. Yet, there are misclassifications where non-BLE signals or noise are labeled as BLE, and occasional BLE signals are mislabeled as other protocols, such as LTE or Z-Wave. This could be due to overlapping frequency bands or the absence of an "unknown" class in the model, forcing it to assign a known protocol label to ambiguous or noisy regions.


![image](https://github.com/user-attachments/assets/3cec5447-6cff-4c76-8ea0-97383f267816)
* The evaluation results for LoRa show that the model confuses most of the LoRa signals with Z-Wave, as reflected in the inference plot. A comparison of the spectrograms for LoRa and Z-Wave reveals that they share similar spectral characteristics, particularly the narrow bandwidth used by both protocols. Also, the chirp patterns of LoRa are not very obvious in the data, making it harder for the model to differentiate between the two protocols. To address this issue, we could improve the dataset by using our own LoRa signals and adjusting the chirp configurations to emphasize the distinctive features of LoRa. This would enable the model to better identify and separate LoRa from Z-Wave signals.

# 5. Discussion and Conclusions

Throughout the project, we have processed collected signal data using binary mask, Fast Fourier Transform, threshold and other methods. Then we generated "stitched", labeled signal bank with different types of signals to train our Deep Learning model to classify signal type on the I/Q level. 

We have also modified the model to be able to classify z-wave signal and trained it with a combination of self-collected and synthetic data from the paper, and adjusted thresholds for each signal type to obtain the optimal performance.

We believe there is huge room for improvement for classification in Wi-Fi and LTE. And in the future, we plan to use our own collected data for BLE, Zigbee, LTE protocols to make the model better reflect real-world situations. We also plan to introduce a new class "unknown" to our model so that noise and signals that do not belong to any of our signal types could be identified as unknown signal, which improves the model's ability to handle real-world scenarios.

# 6. References
[1] Bouchabou, Damien, et al. “A survey of human activity recognition in smart homes based on IoT sensors algorithms: Taxonomies, challenges, and opportunities with deep learning.” Sensors 21.18 (2021): 6037. sensors-21-06037.pdf

[2] Spectro-temporal RF identification using Deep Learning - arXiv. (n.d.). https://arxiv.org/pdf/2107.05114

[3] Uvaydov, D., Zhang, M., Robinson, C. P., D’Oro, S., Melodia, T., & Restuccia, F. (2024a, February 7). Stitching the spectrum: Semantic Spectrum segmentation with wideband signal stitching. arXiv.org. https://arxiv.org/abs/2402.03465

[4] Semantic Spectrum segmentation with wideband signal … (n.d.). https://ece.northeastern.edu/wineslab/papers/UvaydovINFOCOM2024.pdf

[5] Uvaydovd. (n.d.). Uvaydovd/spectrum_sensing_stitching. GitHub. https://github.com/uvaydovd/spectrum_sensing_stitching.git
