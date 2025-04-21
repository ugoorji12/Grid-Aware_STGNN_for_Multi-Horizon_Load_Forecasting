# STT-STGNN-GAT_LSTM for Multi-horizon (1, 6 &amp; 24 hours) Short-term Load Forecasting 

This repository contains the implementation of the Space-then-Time variant Spatiotemporal Graph Neural Network model (STT-STGNN) GAT_LSTM, a hybrid approach that combines multiscale Graph Attention Networks (GAT) and Long Short-Term Memory Networks (LSTM) for multi-horizon short-term power load forecasting. The model leverages the spatio-temporal dependencies in energy systems, incorporating graph-structured data (e.g., power grid topology) and temporal sequences (e.g., historical energy consumption and weather data). **Detailed explanation of the model and results are captured in the paper**: ....

**Features**

i. Graph Attention Network (GAT): Utilizes multiscale (1hop & 2hop) GAT layers to capture spatial relationships and node-level interactions in a graph representing the power grid.

ii. Spatial and Temporal Fusion: Combines the graph-derived embeddings with the sequence (temporal) data before feeding to the sequence processor.

iii. LSTM: Processes the fused, temporally-aware embeddings to make the final hourly power load prediction.


**Data 	Sources**

i. Electricity	(Load, PV, wind, etc.):	https://gitlab.com/dlr-ve/esy/open-brazil-energy-data/open-brazilian-energy-data

ii. Grid	(Line length, capacity, efficiency, etc.): https://gitlab.com/dlr-ve/esy/open-brazil-energy-data/open-brazilian-energy-data

iii. Weather	(Air temperature, pressure, rainfall, etc.): https://www.kaggle.com/datasets/gregoryoliveira/brazil-weather-information-by-inmet?resource=download

iv. Socio-economic	(State-wise GDP & GDP per capita): https://www.ibge.gov.br/en/statistics/economic/national-accounts/16855-regional-accounts-of-brazil.html

v. Population	(State-wise population): https://www.ibge.gov.br/en/statistics/social/population/18448-estimates-of-resident-population-for-municipalities-and-federation-units.html?edicao=28688



