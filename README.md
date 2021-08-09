# Interpretable Deep Learning for Probabilistic MJO Prediction

This repository contains the code used for the paper "Interpretable Deep Learning for Probabilistic MJO Prediction" by A. Delaunay and H. M. Christensen (2021).

## Abstract
<p align="justify">
<strong>The Madden–Julian Oscillation (MJO) is the dominant source of sub-seasonal variability in the tropics.</strong> It consists of an Eastward moving region of enhanced convective storms coupled to changes in zonal winds at the surface and aloft. The chaotic nature of the Earth System means that it is not possible to predict the precise evolution of the MJO beyond a few days, so subseasonal forecasts must be probabilistic.</p>
<p align="justify">  
The forecast probability distribution should vary from day to day, depending on the instantaneous predictability of the MJO6. Operational dynamical subseasonal forecasting models do not have this important property. <strong>Here we show that a statistical model trained using deep-learning can produce skilful state-dependent probabilistic forecasts of the MJO. The statistical model explicitly accounts for intrinsic chaotic uncertainty by predicting the standard deviation about the mean, and model uncertainty using a Monte-Carlo dropout approach.</strong></p>
<p align="justify">
Interpretation of the mean forecasts from the statistical model highlights known MJO mechanisms, providing confidence in the model. <strong>Interpretation of the predicted uncertainty identifies new physical mechanisms governing MJO predictability</strong>. In particular, we find the background state of the Walker Circulation is key for MJO propagation, and not underlying Sea Surface Temperatures as previously assumed.</p>
</p>

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
> <p align="justify">The <strong><i>Analysis</i></strong> folder contains the necessary files to preprocess the S2S reforecasts ".txt" files to obtain a forecast dataframe for each lead time, to compute and compare (Log-score and Error-drop) the CNN and dynamical models. PreprocessingDynamical and AnalysisDynamical must be run once per model to obtain the scores of each model. Then PlotLogScoreEDP and PlotSpreadDiagram plots all the scores on a common plot.</p>

		- AnalysisCNN.py
		- AnalysisDynamical.py
		- CompareFeatures.py
		- PreproprocessingDynamical.py
		- PlotLogScoreEDP.py
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

