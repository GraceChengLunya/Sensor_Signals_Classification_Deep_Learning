# Project Proposal

## 1. Motivation & Objective

Many smart homes are equipped with sensors that collect wireless data about the environment and how people interact with their spaces. Yet, much of this information isn't fully utilized to understand daily activities and routines. By combining this sensor data with common knowledge about how devices like door, motion, and multi sensors are typically used, we can gain deeper insight into occupants' behaviors. Our objective is to develop a system that uses advanced AI models to interpret these wireless signals and access the sensor's location. This will enhance the smart home experience, making it more responsive and intuitive to the needs of its users.

## 2. State of the Art & Its Limitations

You Only Look Once (YOLO) is a current widely used object detection model that divides the input (usually an image) into grids. For signal classification, YOLO divide a signal into fixed-length windows, or bounding boxes. 

Limitations of You Only Look Once (YOLO):
1. This approach requires the creation of an image out of in-phase/quadrature (I/Q) samples, incurring additional latency. 
2. Being a viable method for computer vision tasks, YOLO does not achieve the level of resolution required for wireless signals.
3. Modern wireless signals such as 5G and LoRa can hardly confirm into square bounding boxes which leads to a significant amount of spectrum being incorrectly classified as occupied, leading to poor spectrum efficiency.

![image](https://github.com/user-attachments/assets/53a64b75-943c-4070-b87d-190cf2493493)


## 3. Novelty & Rationale

The approach we found in a paper is called Stitching-the-Spectrum, which "stitches" different signals together to create samples where signals are overlapping and affected by real-world noises and inferences.

Novel dataset generation pipeline that generates large-scale datasets that 
1. contains signals collected Over-the-Air (OTA) and are affected by real-world conditions
2. can be completely labeled
3. low latency 

Novel custom DL algorithm for multi-label multi-class spectrum sensing based on semantic segmentation that
1. operates at the I/Q level instead of creating images
2. classifies each I/Q sample incoming from the ADC without creating bounding boxes, thus increasing classification accuracy significantly
3. only uses 1024 I/Q samples as input, which leads to very low inference time.

![image](https://github.com/user-attachments/assets/59a423ca-e433-4612-89a5-04cfc3a3eea9)


## 4. Potential Impact

Technical impacts:
1. The network is able to achieve more accuracy (7% in the paper) than U-Net in the most challenging protocols (Wi-Fi and LTE) while maintaining similarly low latency
2. The dataset generation pipeline makes the wideband spectrum sensing algorithm more accurate even with data collected from different devices, sampling rates, antennas.
3. Intersection over Union (IoU) of 96.70%; 2.6ms of latency to process 100MHz of spectrum.

Broad impacts:
1. Less amount of data needed for understanding the signal (protocol, MAC address, etc.)
2. A more accurate classification result that takes less time
3. Significantly lower the difficulty of learning about the information of sensors (their types and locations) in a smart home without prior knowledge

## 5. Challenges

1. Data must be collected Over-the-Air (OTA) to capture real-world channel conditions and prepare the AI in a wireless setting with real radios
2. OTA data involves many possible combinations of time an frequency, which significantly increases the complexity in the case of wideband spectrum sensing
3. OTA data requires large portions of spectrum without any interference from external systems
4. Training data that assumes synchronization between sensing and transmission center frequencies results in poor accuracy

## 6. Requirements for Success

What skills and resources are necessary to perform the project?

Resources:
1. SDRs (USRP N210)
2. Smart sensors that use different protocols
3. GPUs

![877D5412-BB41-4833-A95D-E1A05EF7F755](https://github.com/user-attachments/assets/b452bd51-285a-450e-a9b1-3795cb587aa1)
   
Skills:
1. The usage of USRP N210 Software Defined Radio (SDR) for collecting wireless signals
2. Signal Processing skills with GNU Radio or languages like python and MATLAB, etc.
3. Understanding of Deep Learning and RF signals

## 7. Metrics of Success

For protocol classification:
1. Intersection of Union (IoU): Formula: Area of Intersection / Area of Union to indicate the quality of math between the predicted and ground truth boxes. (1 means perfect overlap and 0 means no overlap at all)

For localization:
1. The absolute value of difference between the calculated distance and the actual distance. (The smaller the difference the higher of localization accuracy)

## 8. Execution Plan

1. Collect signals of different protocols with a Bandwidth of around 25MHz centering at 2.4GHz (Wi-Fi, ZigBee) and 916MHz (Z-Wave) transmitted by our sensors of interest
2. Pre-process signals by breaking into shorter signals that are (i) cropped to contain only the actual signal transmission, and (ii) bandpass-filtered to only extract the signal of interest.
3. Convert the data to the frequency domain through a Fast Fourier Transformation (FFT) and prune the components outside of the band of interest.
4. Add the processed data to the signal bank.

![image](https://github.com/user-attachments/assets/ab570a91-5ce4-4522-8107-b60b7748a9b7)


5. Combine multiple signals to generate a "stitched" wideband signal to be added to the training dataset.
6. Train the Deep Learning Model with the training dataset.
7. For each raw IQ signal, we use the corresponding protocol demodulator based on the classification result to demodulate the signal and get usefule information of the packet including the header, address and MAC address.
8. Localize each sensor by calculating the distance between the sensor and the signal receiver based on the change in the Received Signal Strength Indicator (RSSI) incurred by changing the physical location of the receiver.

## 9. Related Work

### 9.a. Papers

1. A Survey of Human Activity Recognition in Smart Homes based on IoT Sensors Algorithms: Taxonomies, Challenges and Opportunities with Deep Learning:

This paper introduces Human Activity Recognition (HAR) in smart home settings, it gave us a overview of the ultimate purpose of our project (Inferring human activities using sensor data and machine learning)

2. Spectro-Temporal RF Identification using Deep Learning:

This paper introduces a deep learning-based system called WRIST for real-time RF signal identification. It employs a detection framework like YOLO to transform RF signals into 2D images for the purpose of detection of multiple signal types. It gave us an overview of how object identification methods in YOLO such as building bounding boxes could be applied to signal classification.

3. Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching:

We decided to use the method introduced in this paper for spectrum sensing. It discussed inefficiency of traditional methods that rely on spectrograms and bounding boxes and introduced a data generation pipeline and a customized U_Net_based deep learning model.


### 9.b. Datasets

External dataset for data of 5 wireless protocols:
https://repository.library.northeastern.edu/collections/neu:h989ss07v

Internet dataset for z-wave data:
https://drive.google.com/file/d/1PwcM4U9z-G7ha2sijdSv-7-N7avlKNQc/view?usp=drive_link
https://drive.google.com/file/d/1kLUZdtNc9FpWMcTaRHV3cT-DRSXEcN6A/view?usp=drive_link
https://drive.google.com/file/d/1kLUZdtNc9FpWMcTaRHV3cT-DRSXEcN6A/view?usp=drive_link


### 9.c. Software

The Stitching-the-Spectrum github project:
https://github.com/uvaydovd/spectrum_sensing_stitching.git

## 10. References

[1] Bouchabou, Damien, et al. "A survey of human activity recognition in smart homes based on IoT sensors algorithms: Taxonomies, challenges, and opportunities with deep learning." Sensors 21.18 (2021): 6037. sensors-21-06037.pdf

[2] Spectro-temporal RF identification using Deep Learning - arXiv. (n.d.). https://arxiv.org/pdf/2107.05114 

[3] Uvaydov, D., Zhang, M., Robinson, C. P., D’Oro, S., Melodia, T., & Restuccia, F. (2024a, February 7). Stitching the spectrum: Semantic Spectrum segmentation with wideband signal stitching. arXiv.org. https://arxiv.org/abs/2402.03465

[4] Semantic Spectrum segmentation with wideband signal ... (n.d.). https://ece.northeastern.edu/wineslab/papers/UvaydovINFOCOM2024.pdf 

[5] Uvaydovd. (n.d.). Uvaydovd/spectrum_sensing_stitching. GitHub. https://github.com/uvaydovd/spectrum_sensing_stitching.git 
