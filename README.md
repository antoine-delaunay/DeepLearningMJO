# Interpretable Deep Learning for Probabilistic MJO Prediction

This repository contains the code used for the paper "Interpretable Deep Learning for Probabilistic MJO Prediction" by A. Delaunay and H. M. Christensen (2022).

## Abstract
<p align="justify">
The Madden–Julian Oscillation (MJO) is the dominant source of sub-seasonal variability in the tropics. It consists of an Eastward moving region of enhanced convection coupled to changes in zonal winds. It is not possible to predict the precise evolution of the MJO, so sub-seasonal forecasts are generally probabilistic.</p>
<p align='justify'>
We present a deep convolutional neural network (CNN) that produces skilful state-dependent probabilistic MJO forecasts. Importantly, the CNN’s forecast uncertainty varies depending on the instantaneous predictability of the MJO. The CNN accounts for intrinsic chaotic uncertainty by predicting the standard deviation about the mean, and model uncertainty using Monte-Carlo dropout. </p>
<p align ='justify'>
Interpretation of the CNN mean forecasts highlights known MJO mechanisms, providing confidence in the model. Interpretation of forecast uncertainty indicates mechanisms governing MJO predictability. In particular, we find an initially stronger MJO signal is associated with more uncertainty, and that MJO predictability is affected by the state of the Walker Circulation.</p>

## Disclaimer
This code is not suitable for a use on a conventional laptop (e.g. memory or CPU capacity). The authors decline any responsibility associated with the use of this repository.

## License
This repository is provided under the GNU General Public License v3.0.

## Dependencies
Packages used in their latest version as of 09/08/2021:
- Python3
- PyTorch
- Numpy
- Pandas
- Matplotlib
- netCDF4
- tqdm
- Cartopy
- Dropblock from https://github.com/miguelvr/dropblock
- PatternNet from https://github.com/TNTLFreiburg/pytorch_patternnet

## Repository structure
**NB: Files directories and other variables must be adapted to your environment and needs in all the scripts**

The repository is organised as follows:
- CNN
>  <p align="justify">The <strong><i>CNN</i></strong>  folder contains the files to build a model and a dataset (once the input fields have been first preprocessed following Wheeler and Hendon) and train the CNN. CompareFeatures compares the performance of the CNN with different subsets of input features and requires to have trained before different CNNs beforehand.</p>
>
		- Dataset.py
		- Model.py
		- Preprocessing.py
		- Train.py
		- CompareFeatures.py 
		
- Analysis
> <p align="justify">The <strong><i>Analysis</i></strong> folder contains the necessary files to preprocess the S2S reforecasts ".txt" files to obtain a forecast dataframe for each lead time, to compute and compare (Log-score and Error-drop) the CNN and dynamical models. PreprocessingDynamical and AnalysisDynamical must be run once per model to obtain the scores of each model. Then PlotMetrics and PlotSpreadDiagram plots all the scores on a common plot.</p>

		- AnalysisCNN.py
		- AnalysisDynamical.py
		- PreproprocessingDynamical.py
		- PlotMetrics.py
		- PlotSpreadDiagram.py
- Interpretation
>  <p align="justify">The <strong><i>Interpretation</i></strong>  folder contains the necessary files to interpret the network's behaviour and the predictability of the forecasts. The signals can be computed by running Train followed by ComputeSignal. The Maritime Continent, Predictability and SignalMeans plots can be run with the associated files but AnalysisCNN must have been run beforehand. </p>

		- MaritimeContinent
			- PlotMC.py
		- PatternNet
			- Train.py
			- ComputeSignal.py
			- patterns.py
			- PatternNetworks.py
			- PatternLayers.py
		- Predictability
			- PlotPredictability.py
		- SignalMeans
			- PlotSignalMeans.py

