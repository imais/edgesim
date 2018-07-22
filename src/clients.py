import logging
import math
import numpy as np
import random
import sys
import latlng_util
from enum import Enum
from scipy.stats import expon

log = logging.getLogger()




class Comm:
	@staticmethod
	def init(dc, sigma1, sigma2, bw_mobile, bw_wan, bw_lan):
		Comm.dc = dc
		Comm.sigma1 = sigma1
		Comm.sigma2 = sigma2
		Comm.bw_mobile = bw_mobile
		Comm.bw_wan = bw_wan
		Comm.bw_lan = bw_lan
		
		
	@staticmethod
	def estimate_dc_comm_time(dc1_id, dc2_id, data):
		dc1 = Comm.dc.loc[dc1_id]
		dc2 = Comm.dc.loc[dc2_id]		
		dist_mi = LatLngUtil.compute_dist_mi(dc1.lat, dc1.lng,
											 dc2.lat, dc2.lng)
		bw = Comm.bw_lan if dc1_id == dc2_id else Comm.bw_wan
		return Comm.sigma1 + Comm.sigma2 * dist_mi + data / bw

		
	@staticmethod
	def estimate_query_comm_time(client_lat, client_lng, dc_id, data):
		dc = Comm.dc.loc[dc_id]
		dist_mi = LatLngUtil.compute_dist_mi(client_lat, client_lng,
											 dc.lat, dc.lng)
		return Comm.sigma1 + Comm.sigma2 * dist_mi + data / Comm.bw_mobile	


class Execute:
	@staticmethod
	def init(dc, beta1, beta2, gamma1, gamma2, theta1, theta2):
		Comm.dc = dc
		Comm.beta1 = beta1
		Comm.beta2 = beta2
		Comm.gamma1 = gamma1
		Comm.gamma2 = gamma2
		Comm.theta1 = theta1
		Comm.theta2 = theta2		

		
	@staticmethod
	def estimate_map_reduce_time(dc_id, data):
		m = Comm.dc.loc[dc_id].m
		return data / (Execute.beta1 + Execute.beta2 * m)

	
	@staticmethod
	def estimate_reduce_time(dc_id, data):
		m = Comm.dc.loc[dc_id].m
		return data / (Execute.gamma1 + Execute.gamma2 * m)

	
	@staticmethod
	def estimate_query_time(dc_id, data):
		m = Comm.dc.loc[dc_id].m
		return data / (Execute.theta1 + Execute.theta2 * m)		


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
	query_data_bytes = 0
	query_resp_bytes = 0

	@staticmethod
	def set_params(query_req_bytes, query_resp_bytes):
		Query.query_req_bytes = query_req_bytes
		Query.query_resp_bytes = query_resp_bytes		
		
	
	def __init__(self, src_lat, src_lng, dest_id):
		self.src_lat = src_lat
		self.src_lng = src_lng
		self.dest_id = dest_id


	def estimate_resp_time(self, query_num):
		self.comm_req_time = Comm.estimate_query_comm_time(self.src_lat, self.src_lng, self.dest_id, Query.query_data_bytes)
		self.query_time = Execute.estimate_query_time(self.dest_id, query_num * Query.query_data_bytes)
		self.comm_resp_time = Comm.estimate_query_comm_time(self.src_lat, self.src_lng, self.dest_id, Query.query_resp_bytes)
		self.total_resp_time = self.comm_req_time + self.query_time + self.comm_resp_time


class QueryClient(object):
	num_queries = {}

	def create_queries(self):
		queries = []

		city_others_indices = self.dc.loc[(self.dc.type == 'city') &
										  (self.dc.index != self.city.name)]
		county = self.dc.loc[self.city.parent_id]
		county_others_indices = self.dc.loc[(self.dc.type == 'county') &
											(self.dc.index != county.name)]
		state =  self.dc.loc[county.parent_id]
		state_others_indices = self.dc.loc[(self.dc.type == 'state') &
										   (self.dc.index != state.name)]
		region = self.dc.loc[state.parent_id]

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
			if dest_id in QueryClient.num_queries:
				QueryClient.num_queries[dest_id] += 1
			else:
				QueryClient.num_queries[dest_id] = 1				
						   
		return queries


	def estimate_query_resp_times(self):
		for query in self.queries:
			query.estimate_resp_time(num_queries[query.dest_id])
			
					 
	def __init__(self, conf, dc, topo, city):
		self.conf = conf
		self.dc = dc
		self.topo = topo
		self.city = city
		self.city_radius = math.sqrt(city.land_area / math.pi)
		self.num_queries_per_min = int(city.population * self.conf['ratio_of_adults'] * \
									   self.conf['ratio_of_adult_smartphone_owners'] * \
									   self.conf['num_queries_per_smartphone_per_hour'] / 60)
		self.queries = create_queries()
