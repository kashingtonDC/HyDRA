# CVWBM

Aakash Ahamed  
Stanford University  
Creation Date: January 2018  

### This repo contains codes for a remote sensing based central valley water balance model, relying upon: 

- Python 3.6
- Earth Engine
- GDAL, Geopandas, Rasterio, Fiona, other geospatial libs
- Numpy, scipy, pandas, other useful python libs
- Tensorflow, Keras, sklearn, other ML libs

This system has been run and tested on MacOS High Sierra 10.13.6

## Build Instructions

### Download Anaconda3
[Link to MacOS Anaconda Installers](https://www.anaconda.com/download/#macos)

### Clone this repository
`git clone https://github.com/kashingtonDC/geospatial_build`

### Create a new conda virtual environment from the env.yml file
```
conda env create -f env.yml
```

This will take a while to install the packages. The name of the environment is 'gis', which is specified in the first line of the env.yml file

Activate the environment:

```
source activate gis
```

Note that for newer versions of conda, this command may be `conda activate gis`. Then (gis) will be prepended to your path. 

### Now install Tensowflow backend:
```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py3-none-any.whl
```

### And Keras:
```
pip install keras
```

### Test everything:
```
python
import ee
import gdal
import tensorflow as tf
import keras
ee.Initialize() # Supply your EE credentials
```

That's it!