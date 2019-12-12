# Earthquakes

The earthquake classification problem involves predicting whether a major event is about to occur based on the most recent readings in the surrounding area. The data are taken from Northern California Earthquake Data Center and each data point is an averaged reading for one hour, with the first reading taken on Dec 1st 1967 and the last in 2003. This single time series are then turned into a classification problem of differentiating between a positive and negative major earthquake event.

A major event is defined as any reading of over 5 on the Rictor scale. Major events are often followed by aftershocks. (The physics of these are well understood and their detection is not the objective of this dataset.) A positive case is defined a major event which is not preceded by another major event for at least 512 hours. 

Negative cases are instances where there is a reading below 4 (to avoid blurring of the boundaries between major and non-major events) that is preceded by at least 20 non-zero readings in the previous 512 hours (to avoid trivial negative cases). 

In total, 368 negative and 93 positive cases were extracted from 86,066 hourly readings. None of the cases overlap in time (i.e. a segmentation is used instead of a sliding window).

Train size: 322

Test size: 139

Missing value: No

Number of classses: 2

Time series length: 512

Data donated by Anthony Bagnall (see [1]).

[1] http://www.timeseriesclassification.com/description.php?Dataset=Earthquakes
