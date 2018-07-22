import logging
import numpy as np
import random
import sys
import util
from enum import Enum
from scipy.stats import expon

log = logging.getLogger()


class QueryDestType(Enum):
	NULL = 0
	CITY_SELF = 1
	CITY_OTHER = 2
	COUNTY_SELF = 3
	COUNTY_OTHER = 4
	STATE_SELF = 5
	STATE_OTHER = 6
	REGION = 7

	def __str__(self):
		return self.name


class Query(object):
	def __init__(self, src_lat, src_lng, dest_id):
		self.src_lat = src_lat
		self.src_lng = src_lng
		self.dest_id = dest_id


	def compute_query_time(self, total_query_num):
		return query_time



class QueryClient(object):
	destinations = {}

	def get_rnd_lat_lng(self):
		bearing = random.uniform(0, 1) * 360
		dist_mi = random.uniform(0, 1) * self.city_radius
		return util.compute_lat_lng(self.src_city.lat, self.src_city.lng, bearing, dist_mi)
		

	def create_queries(self):
		queries = []

		city_others_indices = self.geo_db.loc[(self.geo_db.type == 'city') &
											  (self.geo_db.index != self.city.name)]
		county = self.geo_db.loc[self.city.parent_id]
		county_others_indices = self.geo_db.loc[(self.geo_db.type == 'county') &
												(self.geo_db.index != county.name)]
		state =  self.geo_db.loc[county.parent_id]
		state_others_indices = self.geo_db.loc[(self.geo_db.type == 'state') &
												(self.geo_db.index != state.name)]
		region = self.geo_db.loc[state.parent_id]

		for i in range(self.num_queries_per_min):
			r = random.uniform(0, 1)
			total = 0
			dest_type = QueryDestType.CITY_SELF
			while self.conf['query_dist'][dest_type] < r:
				dest_type += 1

			(src_lat, src_lng) = self.get_rnd_lat_lng()				
			if dest_type == QueryDestType.CITY_SELF:
				dest_id = self.city.name
			elif dest_type == QueryDestType.CITY_OTHER:
				dest_id = random.choice(city_others_indices)
			elif dest_type == QueryDestType.COUNTY_SELF:
				dest_id = county.name
			elif dest_type == QueryDestType.COUNTY_OTHER:
				dest_id = random.choice(county_others_indices)
			elif dest_type == QueryDestType.STATE_SELF:
				dest_id = state.name
			elif dest_type == QueryDestType.STATE_OTHER:
				dest_id = random.choice(state_others_indices)
			else:
				# dest_type == QueryDestType.REGION
				dest_id = region.name

			queries.append(Query(src_lat, src_lng, dest_id))
			if dest_id in QueryClient.destinations:
				QueryClient.destinations[dest_id] += 1
			else:
				QueryClient.destinations[dest_id] = 1				
						   
		return queries


	def compute_resp_time(self):
		# for query in self.queries:
		return
			
					 
	def __init__(self, conf, geo_db, city):
		self.conf = conf
		self.geo_db = geo_db
		self.city = city
		self.city_radius = math.sqrt(city.land_area / math.pi)
		self.num_queries_per_min = int(city.population * self.conf['ratio_of_adults'] * \
									   self.conf['ratio_of_adult_smartphone_owners'] * \
									   self.conf['num_queries_per_smartphone_per_hour'] / 60)
		self.comm_params = self.conf['comm_params']
		self.comm_params = self.conf['comm_params']
		
		self.queries = create_queries()
