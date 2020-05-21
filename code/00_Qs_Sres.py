#!/usr/bin/env python
# coding: utf-8

# This script retrieves surface water data for the Central Valley and Central Valley Watershed. 

import os
import geopandas as gp
import pandas as pd
import numpy as np
import shapely
import datetime
import io
import requests
import urllib.request

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from climata.usgs import DailyValueIO
from shapely.geometry import Point
from shapely.ops import cascaded_union


# Helpers ---------------------------------------------------

def fetch_nhd(huc):
	'''
	Given a huc4, fetch the Nhd dataset from the USGS, unzip it, and put it in the ../data folder
	'''
	nhddir = "../nhd"
	outdir = os.path.join(nhddir,huc) 
	outfile = os.path.join(nhddir,huc + ".zip") 

	if os.path.exists(outdir):
		pass
	else:
		cmd = '''curl -o {}  https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU4/HighResolution/Shape/NHD_H_{}_HU4_Shape.zip && mkdir {} && tar -xvf {} && mv Shape/ {}/ && rm {}'''.format(outfile, huc,outdir, outfile, outdir, outfile)
		os.system(cmd)


def find_hucs(dir, huc_str):
	'''
	Find the directories 
	'''

	shp_files = [os.path.join(dir,x) for x in os.listdir(dir) if huc_str in x if "xml" not in x]
	return shp_files

# Main functions ---------------------------------------------------

def get_streamflow(huc8):
	'''
	call climata API supplying huc8 argument to get each gaging station within each basin 
	'''
	
	data =  DailyValueIO (
			start_date="2001-01-01", 
			end_date="2018-01-01",
			basin=huc8,
			parameter="00060",
			)
	
	qs = []
	ds = []
	lats = []
	lons = []
	ids = []

	for series in data:
		values = []
		dates = []
		lats.append(series.latitude)
		lons.append(series.longitude)
		ids.append(series.site_code)

		for row in series.data:
			values.append(row.value)
			dates.append(row.date)

		qs.append(values)
		ds.append(dates)
	
	geometry = [Point(xy) for xy in zip(lons, lats)]
	df = pd.DataFrame(geometry)
	crs = {'init': 'epsg:4326'}
	gdf = gp.GeoDataFrame(df, crs=crs, geometry=geometry)
	
	return gdf, qs, ds, ids


def streamflow():

	print("**** Begin Fetching USGS Streamflow Data ****")

	huc_order = "8"
	huc_str = "WBDHU{}.shp".format(huc_order)
	nhddir = "../nhd"

	if not os.path.exists(nhddir):
		os.mkdir(nhddir)

	huc4s = ["1802", "1803", "1804"]
	for i in huc4s: 
		fetch_nhd(i)

	hu4_dirs = [os.path.join("../nhd", x, "Shape") for x in os.listdir("../nhd") if "." not in x]


	gdfs = []

	for i in hu4_dirs:
		gdfs.append(gp.read_file(find_hucs(i, huc_str)[0]))

	# Concat the HU4s to form the CV watershed. 
	hu4 = gp.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

	if not os.path.exists("../shape/cvws.shp"):
		hu4.to_file("../shape/cvws.shp")
		cvws = hu4.copy()

	else:
		cvws = gp.read_file("../shape/cvws.shp")

	
	# For each HU4, go in , grab the hu8s, and use these to query the Climata API for streamflow. 
	gdfs = []

	for i in hu4_dirs:
		gdfs.append(gp.read_file(find_hucs(i, huc_str)[0]))

	hu4 = gp.GeoDataFrame(pd.concat(gdfs, ignore_index=True))


	gdfs = []
	qs = []
	ds = []
	ids = []

	for i in hu4['HUC8']:
		print ("processing " + i)
		gdf, q, d, i = get_streamflow(i)
		gdfs.append(gdf)
		qs.append(q)
		ds.append(d)
		ids.append(i)


	# Un-nest the lists
	q_f = np.array([item for sublist in qs for item in sublist])
	d_f = np.array([item for sublist in ds for item in sublist])
	ids_f = [item for sublist in ids for item in sublist]


	# Make a gdf of the stations and join the lists of q, ID and dates
	stations_gdf = gp.GeoDataFrame(pd.concat(gdfs, ignore_index=True, sort = False))
	stations_gdf['Q'] = q_f
	stations_gdf['ID'] = ids_f
	stations_gdf['dates'] = d_f

	# Inflows from xiao et al (2017)+ ones I added in last row 
	stations = [11446500, 11376550, 11423800, 11384000, 11390000 ,11451760,
				11372000, 11335000, 11376000, 11374000, 11383500, 11329500,
				11211300, 11424500, 11379500, 11407150, 11257500, 11209900,
				11192950, 11251600, 11225000, 11270900, 11381500, 11221700,
				11325500, 11384350, 11454000, 11370500, 11251000, 11302000, 
				11388000, 11382000, 11289650, 11199500, 11421000, 
			   
				11208818, 11204100, 11200800, 11218400, 11289000, 11323500
			   ]

	# The CA Aqueduct takes water out of the CV: 
	stations_out = ["11109396"]

	stations = [str(x) for x in stations]

	# Separate the inflows / outflows 
	inflow = stations_gdf[stations_gdf['ID'].isin(stations)]
	outflow = stations_gdf[stations_gdf.ID == "11109396"]

	in_dfs = []

	for idx, x in enumerate(q_f):
		sdf1 = pd.DataFrame(q_f[idx], d_f[idx], columns = [ids_f[idx]])
		in_dfs.append(sdf1)

	# Filter for the stations of interest
	fin = pd.concat(in_dfs, axis = 1)

	fin_in = fin.loc[:, fin.columns.str.contains('|'.join(stations))]

	# calc the daily sums in, subtract the CA aqueduct outflow. When outflow is nan use mean outflow 
	fin_in['sum_cfs'] = fin_in.sum(axis = 1) - fin["11109396"].fillna(fin["11109396"].mean())

	# Calc the monthly sums
	sum_df = pd.DataFrame(fin_in['sum_cfs'].resample('M').sum())

	# Convert cfs to km3 / mon
	sum_df['sum_km3'] = sum_df.sum_cfs * 0.0283168 * 1e-9 * 86400 # convert CFS to cms (0.0283168), CMS to km^3 /s (1e-9) , km^3/s to km^3 / mon (86400)
	sum_df.to_csv("../data/Qs_in_monthly.csv")

	print('average total streamflow volume = {} km^3'.format(np.mean(sum_df)))


	return(stations)


def dayflow():

	print("**** Begin Fetching Outflow to SF bay (DWR Dayflow) ****")

	dayflow_path = '../data/dayflow-results-1997-2019.csv'
	dayflow_url = 'https://data.cnra.ca.gov/dataset/06ee2016-b138-47d7-9e85-f46fae674536/resource/21c377fe-53b8-4bd6-9e1f-2025221be095/download/dayflow-results-1997-2019.csv?accessType=Download'
	if not os.path.exists(dayflow_path):
		cmd = '''curl -L -o {} {}'''.format(dayflow_path, dayflow_url)
		print(cmd)
		os.system(cmd)

	dayflow = pd.read_csv(dayflow_path)
	dayflow.index = pd.to_datetime(dayflow.Date)
	monthly = dayflow.OUT.resample("M").sum() * 0.0283168 * 1e-9 * 86400 # convert CFS to cms (0.0283168), CMS to km^3 /s (1e-9) , km^3/s to km^3 / mon (86400)
	print('average monthly SF bay outflow = {} km^3'.format(np.mean(monthly)))
	monthly.to_csv("../data/Qs_out_monthly.csv")
	
	print("Outflow to SF Bay DONE ====================================== ")

	return(monthly)


def res_storage(shp, outfn):

	print("**** Begin Fetching CDEC Reservoir Storage Data for {} ****".format(shp))

	# Read the shapefile 
	gdf = gp.read_file(shp)

	# Spatial join cdec reservoirs to supplied gdf 
	reservoirs = gp.read_file("../shape/cdec_reservoirs.shp")
	within_gdf = gp.sjoin(reservoirs, gdf, how='inner', op='within')
	
	# Download Storage (SensorNums = 15) data by query str:
	start = datetime.datetime(2001, 1, 1)
	end = datetime.datetime(2019, 1, 1)
	dt_idx = pd.date_range(start,end, freq='M')

	data = {}

	for i in within_gdf.ID:
		print("processing " + i )
		url = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={}&SensorNums=15&dur_code=M&Start=2001-01-01&End=2018-12-01".format(i)
		urlData = requests.get(url).content
		df = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
		
		if df.empty:
			pass
		else:
			data[i] = df

	storage = []
	for k,v in data.items():
		storage.append(pd.to_numeric(data[k].VALUE, errors = "coerce"))

	storage_sum = np.nansum(np.column_stack(storage), axis = 1) * 1.23348e-6 # acre ft to km^3
	Sres = pd.DataFrame(zip(dt_idx,storage_sum), columns = ['date',"Sres"])

	Sres.to_csv(outfn)

	print("Mean reservoir storage = {} km^3".format(np.mean(Sres)))

	print("Reservoir Storage DONE ====================================== ")

	return Sres

def main():
	# Required global params: Nhd directory, shapefiles for CV, CVWS, reservoir locations 
	nhddir = "../nhd"
	cv_shp = '../shape/cv.shp'
	cvws_shp = '../shape/cvws.shp'

	# Globals
	cv = gp.read_file(cv_shp)
	cvws = gp.read_file(cvws_shp)
	reservoirs = gp.read_file("../shape/cdec_reservoirs.shp")

	# Main routines 
	cv_Sres = res_storage(cv_shp, "../data/cv_Sres.csv")
	cvws_Sres = res_storage(cvws_shp, "../data/cvws_Sres.csv")
	dayf = dayflow()
	sf = streamflow()

if __name__ == '__main__':
	main()