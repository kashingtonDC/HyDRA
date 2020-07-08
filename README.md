# Hybrid Data Remote Sensing Assimilation (HyDRA) System 

* Author: Aakash Ahamed (aahamed@stanford.edu), Stanford Univ Dept of Geophysics 
* Date: 7/2020
* Description: This repository contains the code and data to perform monthly water balance calculations using remote sensing and in situ data for the Central Valley (CV) and Central Valley Watershed (CVWS) of California. Files describing the geographic domain are stored in the `shape` directory. Scripts to download and process data, and generate figures and results are stored in the `code` directory. Instructions to install the required software dependencies are in the `build` directory. Figures and images are stored in the `images` directory. Below is a list of URLS containing additional datasets for the CV and CVWS. 

<img src="/images/Figure1.png" width="48"> <br>
Study domain

![](/images/cvws_et.gif) <br> 
SSEBop ET in the CVWS

<img src="/images/Figure2.png" width="48"> <br>
Changes in Groundwater Storage


# More information about data used or queried in this repository : 

## Remote Sensing and Satellite Data - Various sources
[Google Earth Engine](https://developers.google.com/earth-engine/datasets/)

## SSEBOP ET data - USGS 
[SSEBOP](https://cida.usgs.gov/thredds/catalog.html?dataset=cida.usgs.gov/ssebopeta/monthly)

[SSEBOP bbox url](https://cida.usgs.gov/thredds/ncss/ssebopeta/monthly/dataset.html)

[SSEBOP.nc](https://cida.usgs.gov/thredds/ncss/ssebopeta/monthly?var=et&north=42.003728&west=-123.217338&east=-117.959444&south=34.459646&horizStride=1&time_start=2000-01-01T00%3A00%3A00Z&time_end=2019-10-01T00%3A00%3A00Z&timeStride=1&addLatLon=true)

## Watershed Data - National Hydrographic Dataset - USGS
[https://www.usgs.gov/core-science-systems/ngp/national-hydrography/nhdplus-high-resolution](https://www.usgs.gov/core-science-systems/ngp/national-hydrography/nhdplus-high-resolution)
```
mkdir nhd
cd nhd

curl -o 1802.zip https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU4/HighResolution/Shape/NHD_H_1802_HU4_Shape.zip && mkdir 1802 && tar -xvf 1802.zip && mv Shape/ 1802/ && rm 1802.zip

curl -o 1803.zip https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU4/HighResolution/Shape/NHD_H_1803_HU4_Shape.zip && mkdir 1803 && tar -xvf 1803.zip && mv Shape/ 1803/ && rm 1803.zip

curl -o 1804.zip https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU4/HighResolution/Shape/NHD_H_1804_HU4_Shape.zip && mkdir 1804 && tar -xvf 1804.zip && mv Shape/ 1804/ && rm 1804.zip

```

## C2VSIM flow model - DWR

click link to download:

[https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/richpauloo/pred_gws/tree/master/data/C2VSimFG-BETA_GIS/Shapefiles](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/richpauloo/pred_gws/tree/master/data/C2VSimFG-BETA_GIS/Shapefiles)

```
mkdir c2vsim
cd c2vsim

mv ~/Downloads/Shapefiles.zip .
tar -xvf Shapfiles.zip
cd Shapefiles
mv * ..
rm -rf Shapefiles Shapefiles.zip

```

## Crop Coefficients - USGS
[https://water.usgs.gov/GIS/dsdl/pp1766_fmp_parameters.zip](https://water.usgs.gov/GIS/dsdl/pp1766_fmp_parameters.zip)

## SF Bay Outflow - DWR
[dayflow](https://water.ca.gov/Programs/Environmental-Services/Compliance-Monitoring-And-Assessment/Dayflow-Data)

## Well data - DWR Casgem 
[casgem](https://data.cnra.ca.gov/dataset/periodic-groundwater-level-measurements)

## CVHM Texture model - USGS
[cvhm texture data](https://ca.water.usgs.gov/projects/central-valley/well-log-texture.xls)

## Reservoir Data - California Data Exchange Center
[http://cdec.water.ca.gov/misc/monthlyStations.html](http://cdec.water.ca.gov/misc/monthlyStations.html) 


# Additional data: 


## Detailed Analysis Units - DWR
[https://github.com/CSTARS/dwr-dau](https://github.com/CSTARS/dwr-dau)

## Applied water, ET - CALSIM DWR
[https://data.ca.gov/dataset/cal-simetaw-unit-values](https://data.ca.gov/dataset/cal-simetaw-unit-values)

## Crop Areas - DWR
[https://data.cnra.ca.gov/dataset/crop-mapping-2014](https://data.cnra.ca.gov/dataset/crop-mapping-2014)
[https://opendata.arcgis.com/datasets/f4f00986f1e141cc99cff221391084de_0.zip](https://opendata.arcgis.com/datasets/f4f00986f1e141cc99cff221391084de_0.zip)


## Aqueducts - DWR 
[https://data.ca.gov/dataset/canals-and-aqueducts-local](https://data.ca.gov/dataset/canals-and-aqueducts-local)

## SGMA GW Basins 
[basin priority data](https://data.cnra.ca.gov/dataset/sgma-basin-prioritization-2018/resource/7bfe794b-b64e-46ee-9d7f-2ca9593cfee2)

## CVHM
[faunt, 2009 model files](https://water.usgs.gov/GIS/dsdl/gwmodels/PP2009-1766/model.zip)

## Major Rivers - CNRA
[major rivers](https://data.cnra.ca.gov/dataset/national-hydrography-dataset-nhd/resource/510abd22-f63b-4981-a17e-3c76cec5fa18)

## Canals / Aqueducts - DWR
[canals and aqueducts](http://atlas-dwr.opendata.arcgis.com/datasets/b788fb2628844f54b92e46dac5bb7229_0)

## State Water Project - USGS NHD
[SWP](https://services7.arcgis.com/x74yAepfzbQsthyi/arcgis/rest/services/NHD_SWP_Aqueduct/FeatureServer/0?f=pjson)
```
curl -o swp.json https://services7.arcgis.com/x74yAepfzbQsthyi/arcgis/rest/services/NHD_SWP_Aqueduct/FeatureServer/0?f=pjson 
ogr2ogr -f "ESRI Shapefile" SWP_Canals.shp swp.json
```
## SW deliveries - USBR
[sw deliveries](https://www.usbr.gov/mp/cvo/deliv.html)