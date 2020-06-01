#!/usr/bin/env python
# coding: utf-8

# This script retrieves surface water data for the Central Valley and Central Valley Watershed. It writes shapefiles to "shape" and results to "data"

import os
import geopandas as gp
import pandas as pd
import numpy as np
import datetime
import io
import requests
import urllib.request

from tqdm import tqdm
from functools import reduce
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
			start_date="1997-01-01", 
			end_date="2020-01-01",
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

	# Loop through the subwatersheds, and extract the streamflow data 
	gdfs = []
	daily_dfs = []

	for i in tqdm(hu4['HUC8'][:]):
		
		gauge_id = i

		# Call streamflow getter
		gdf, q, d, gid = get_streamflow(gauge_id)
		gdf['gauge_id'] = np.array(gid)
		
		print("Processing {} stations for HUC {}".format(str(len(gid)),gauge_id))

		if len(q) == 0 & len(d) == 0: 
			print('no streamflow data for HUC {}'.format(gauge_id))
			continue
		
		# loop through the gauge stations in that huc and build a dataframe 
		hucdfs = []
		for idx, x in enumerate(gid):
			flows = np.array(q[idx]) * 0.0283168 # convert cfs to cms
			dates = d[idx]
			sdf = pd.DataFrame(flows, dates)
			sdf.columns = [x]
			hucdfs.append(sdf)
		
		hucdf = reduce(lambda x, y: pd.merge(x, y, how = 'outer', left_index = True, right_index = True), hucdfs)
		daily_dfs.append(hucdf)
		gdfs.append(gdf)

	# Compile the dfs and gdfs 

	# The site data
	master_df = reduce(lambda x, y: pd.merge(x, y, how = 'outer', left_index = True, right_index = True), daily_dfs)
	master_df.to_csv("../data/daily_q_data.csv")

	# The geographic data  
	stations_gdf = gp.GeoDataFrame(pd.concat(gdfs, ignore_index=True, sort = False))

	# Compute the mean and variance for each site 
	outdfs = []

	for i in master_df.columns.unique():
		monthlydata = master_df[i].resample("M").mean()* 2.628e+6 * 1e-9 # convert seconds to months, m^3 to km^3
		mean = np.mean(monthlydata) 
		var = np.var(monthlydata) 
		start =  np.array(monthlydata.index[0].strftime("%Y-%m-%d %H:%M:%S"))
		end = np.array(monthlydata.index[-1].strftime("%Y-%m-%d %H:%M:%S"))
		sdf = pd.DataFrame([str(i), mean, var,str(start),str(end)]).T
		sdf.columns = ["gauge_id","q_km3_avg", "q_km3_var", "startdate", "enddate"]
		sdf2 = sdf.astype({"gauge_id": str, "q_km3_avg": float, "q_km3_var": float, "startdate": str, "enddate": str})

		outdfs.append(sdf2)

	monthly_means = pd.concat(outdfs)
	mgdf = pd.merge(stations_gdf, monthly_means, left_on = "gauge_id", right_on = "gauge_id")
	fingdf = mgdf.drop([mgdf.columns[0]],axis = 1)

	fingdf.to_file("../shape/usgs_gauges.shp")

	# Inflows from xiao et al (2017)+ ones I added in last row 
	stations = [11446500, 11376550, 11423800, 11384000, 11390000 ,11451760,
				11372000, 11335000, 11376000, 11374000, 11383500, 11329500,
				11211300, 11424500, 11379500, 11407150, 11257500, 11209900,
				11192950, 11251600, 11225000, 11270900, 11381500, 11221700,
				11325500, 11384350, 11454000, 11370500, 11251000, 11302000, 
				11388000, 11382000, 11289650, 11199500, 11421000, 
			   
				11208818, 11204100, 11200800, 11218400, 11289000, 11323500
			   ]
	stations = [str(x) for x in stations]

	# for gdf of inflow stations 
	inflow_gdf = mgdf[mgdf['gauge_id'].isin(stations)]

	# The CA Aqueduct takes water out of the CV: 
	stations_out = ["11109396"]

	inflow = []
	outflow = []

	# Separate the inflows / outflows 
	for i in master_df.columns:
		if i in stations:
			inflow.append(master_df[i].resample("M").mean()*2.628e+6 * 1e-9)# convert seconds to months, m^3 to km^3
		if i in stations_out:
			outflow.append(master_df[i].resample("M").mean()*2.628e+6 * 1e-9)# convert seconds to months, m^3 to km^3

	inflow_df = pd.concat(inflow, axis = 1)
	outflow_df = pd.concat(outflow)

	inflow_sum = inflow_df.sum(axis =1)
	outflow_sum = outflow_df.sum()

	# Calculate the net inflow. For months with no data, replace with mean 
	net_flow = inflow_sum - outflow_df.fillna(outflow_df.mean())

	print("mean monthly inflow = {}".format(np.mean(net_flow)))

	# Write 
	nfdf = pd.DataFrame(net_flow)
	nfdf.columns = ['net_inflow_km3']

	net_flow.to_csv("../data/Qs_in_monthly.csv")

	return(net_flow)


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
	start = datetime.datetime(1997, 1, 1)
	end = datetime.datetime(2020, 1, 1)
	dt_idx = pd.date_range(start,end, freq='M')

	data = {}

	for i in tqdm(within_gdf.ID):
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
	netflow = sf - dayf

	netflow.to_csv("../data/net_flow_monthly.csv")

	print("Complete =======" * 20 )

if __name__ == '__main__':
	main()