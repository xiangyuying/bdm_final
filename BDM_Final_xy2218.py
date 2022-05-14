


import csv
import numpy as np
import pandas as pd
import IPython


import shapely
import pyspark
from pyspark.sql.types import *
from shapely.geometry import Point
from pyproj import Transformer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)
spark

date_range = F.udf(lambda x : x[:7])
def convert_x(x,y):
    t = Transformer.from_crs(4326, 2263)
    return list(t.transform(x,y))[0]

def convert_y(x,y):
    t = Transformer.from_crs(4326, 2263)
    return list(t.transform(x,y))[1]
convert_x = F.udf(convert_x)
convert_y = F.udf(convert_y)

def distance(x1,y1,x2,y2,c):
    return (((x1-x2)**2 + (y1-y2)**2)**0.5/5280)*c
distance = F.udf(distance)  

def avg(x,y):
    return x/y
avg = F.udf(avg)

pattern = spark.read.csv('weekly-patterns-nyc-2019-2020-sample.csv', header=True)\
.select('placekey','poi_cbg','visitor_home_cbgs',date_range('date_range_start'),date_range('date_range_end'))\
.withColumnRenamed('<lambda>(date_range_start)','start').withColumnRenamed('<lambda>(date_range_end)','end')
super = spark.read.csv('nyc_supermarkets.csv', header=True).select('safegraph_placekey').withColumnRenamed('safegraph_placekey','placekey')
cbg = spark.read.csv('nyc_cbg_centroids.csv', header=True)
pattern1 = pattern.join(super, how='inner',on ='placekey')

pattern = pattern1.filter((pattern1['start']=='2019-03')|(pattern1['end']=='2019-03'))
pattern = pattern.filter(pattern['visitor_home_cbgs']!='{}')
pattern = pattern.withColumn('counts',pattern['visitor_home_cbgs'][20:21].cast('int')).dropna()
pattern = pattern.withColumn('home_cbgs',pattern['visitor_home_cbgs'][5:12])
pattern = pattern.join(cbg, pattern.home_cbgs==cbg.cbg_fips , how = 'inner').select('poi_cbg','home_cbgs','counts','latitude','longitude')\
.withColumnRenamed('latitude','home_latitude').withColumnRenamed('longitude','home_longitude')
pattern = pattern.join(cbg, pattern.poi_cbg==cbg.cbg_fips , how = 'inner').drop('cbg_fips')
pattern = pattern.withColumn('home_x',convert_x('home_latitude', 'home_longitude').cast('Float'))
pattern = pattern.withColumn('poi_x',convert_x('latitude', 'longitude').cast('Float'))
pattern = pattern.withColumn('home_y',convert_y('home_latitude', 'home_longitude').cast('Float'))
pattern = pattern.withColumn('poi_y',convert_y('latitude', 'longitude').cast('Float'))
pattern = pattern.withColumn('distance',distance('home_x','home_y','poi_x','poi_y','counts').cast('Float'))
pattern = pattern.groupby('home_cbgs').sum('counts','distance')
final = pattern.withColumn('2019-03',avg('sum(distance)','sum(counts)')).drop('sum(distance)','sum(counts)')

for i in ['2019-10','2020-03','2020-10']:
    pattern = pattern1.filter((pattern1['start']==i)|(pattern1['end']==i))
    pattern = pattern.filter(pattern['visitor_home_cbgs']!='{}')
    pattern = pattern.withColumn('counts',pattern['visitor_home_cbgs'][20:21].cast('int')).dropna()
    pattern = pattern.withColumn('home_cbgs',pattern['visitor_home_cbgs'][5:12])
    pattern = pattern.join(cbg, pattern.home_cbgs==cbg.cbg_fips , how='inner')\
    .select('poi_cbg','home_cbgs','counts','latitude','longitude')\
    .withColumnRenamed('latitude','home_latitude').withColumnRenamed('longitude','home_longitude')
    pattern = pattern.join(cbg, pattern.poi_cbg==cbg.cbg_fips , how = 'inner').drop('cbg_fips')
    pattern = pattern.withColumn('home_x',convert_x('home_latitude', 'home_longitude').cast('Float'))
    pattern = pattern.withColumn('poi_x',convert_x('latitude', 'longitude').cast('Float'))
    pattern = pattern.withColumn('home_y',convert_y('home_latitude', 'home_longitude').cast('Float'))
    pattern = pattern.withColumn('poi_y',convert_y('latitude', 'longitude').cast('Float'))
    pattern = pattern.withColumn('distance',distance('home_x','home_y','poi_x','poi_y','counts').cast('Float'))
    pattern = pattern.groupby('home_cbgs').sum('counts','distance')
    pattern = pattern.withColumn(i,avg('sum(distance)','sum(counts)')).drop('sum(distance)','sum(counts)')
    final = final.join(pattern, 'home_cbgs', 'full')

final.saveAsTextFile('final')

