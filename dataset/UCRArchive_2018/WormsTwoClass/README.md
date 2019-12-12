# WormTwoClass

Caenorhabditis elegans (C. elegans) is a roundworm commonly used as a model organism in genetics study. The movement of these worms is known to be a useful indicator for understanding behavioural genetics. 

The data were original from [1][2], in which the authors described a system for recording the motion of worms on an agar plate and measuring a range of human-defined features. 

It has been shown that the space of shapes Caenorhabditis elegans adopts on an agar plate can be represented by combinations of four base shapes, or eigenworms. Once the worm outline is extracted, each frame of worm motion can be captured by four scalars representing the amplitudes along each dimension when the shape is projected onto the four eigenworms. 

The data were formatted for time series classification task and used in [3]. Each case is a series of the first eigenworm only, down-sampled to second-long intervals and averaged down so that all series are of length 900. There are 258 cases in total; each belongs to one of five types: one wild-type (the N2 reference strain - 109 cases) and four mutants: goa-1 (44 cases), unc-1 (35 cases), unc-38 (45 cases) and unc-63 (25 cases). 

In case of the *WormsTwoClass* dataset, the task is to classify worms of wild-type or mutant-type.

In case of the *Worms* dataset, the task is to classify worms into one of the five categories. 

Train size: 181

Test size: 77

Missing value: No

Number of classses: 2

Time series length: 900

Data donated by Andre Brown and Anthony Bagnall (see [1], [3]).

[1] Brown, André EX, et al. "A dictionary of behavioral motifs reveals clusters of genes affecting Caenorhabditis elegans locomotion." Proceedings of the National Academy of Sciences 110.2 (2013): 791-796.

[2] Yemini, Eviatar, et al. "A database of Caenorhabditis elegans behavioral phenotypes." Nature methods 10.9 (2013): 877.

[3] Bagnall, Anthony, et al. "Time-series classification with COTE: the collective of transformation-based ensembles." IEEE Transactions on Knowledge and Data Engineering 27.9 (2015): 2522-2535.

[4] http://www.timeseriesclassification.com/description.php?Dataset=Worms