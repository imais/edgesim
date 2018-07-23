import logging
import math
import numpy as np
import random
import sys
from latlng_util import LatLngUtil
from enum import IntEnum
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


class Exec:
	@staticmethod
	def init(dc, beta1, beta2, gamma1, gamma2, theta1, theta2):
		Exec.dc = dc
		Exec.beta1 = beta1
		Exec.beta2 = beta2
		Exec.gamma1 = gamma1
		Exec.gamma2 = gamma2
		Exec.theta1 = theta1
		Exec.theta2 = theta2		

		
	@staticmethod
	def estimate_map_reduce_time(dc_id, data):
		m = Exec.dc.loc[dc_id].m
		return data / (Exec.beta1 + Exec.beta2 * m)

	
	@staticmethod
	def estimate_reduce_time(dc_id, data):
		m = Exec.dc.loc[dc_id].m
		return data / (Exec.gamma1 + Exec.gamma2 * m)

	
	@staticmethod
	def estimate_query_time(dc_id, data):
		m = Exec.dc.loc[dc_id].m
		return data / (Exec.theta1 + Exec.theta2 * m)		


class QueryDestType(IntEnum):
	CITY_SELF = 0
	CITY_OTHER = 1
	COUNTY_SELF = 2
	COUNTY_OTHER = 3
	STATE_SELF = 4
	STATE_OTHER = 5
	REGION = 6
	UNKNOWN = 7

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
		self.query_time = Exec.estimate_query_time(self.dest_id, query_num * Query.query_data_bytes)
		self.comm_resp_time = Comm.estimate_query_comm_time(self.src_lat, self.src_lng, self.dest_id, Query.query_resp_bytes)
		self.total_resp_time = self.comm_req_time + self.query_time + self.comm_resp_time


class QueryClient(object):
	num_queries = {}

	def create_queries(self):
		self.queries = []

		city_others_indices = self.dc.loc[(self.dc.type == 'city') &
										  (self.dc.index != self.city.name)].index

		county = self.dc.loc[self.city.parent_id]
		county_others_indices = self.dc.loc[(self.dc.type == 'county') &
											(self.dc.index != county.name)].index
		state =  self.dc.loc[county.parent_id]
		state_others_indices = self.dc.loc[(self.dc.type == 'state') &
										   (self.dc.index != state.name)].index

		for i in range(self.num_queries_per_min):
			r = random.uniform(0, 1)
			dest_type = QueryDestType.CITY_SELF			
			total = self.conf['query_dist'][int(dest_type)]
			# query_dist: [city_self, city_others, county_self, county_others, state_self, state_others, region]
			while total < r:
				dest_type += 1
				total += self.conf['query_dist'][int(dest_type)]				

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
				dest_id = self.dc.loc[state.parent_id].name

			(src_lat, src_lng) = LatLngUtil.get_rand_lat_lng(self.city.lat, self.city.lng,
															 360, self.city_radius)
			self.queries.append(Query(src_lat, src_lng, dest_id))
			
			if dest_id in QueryClient.num_queries:
				QueryClient.num_queries[dest_id] += 1
			else:
				QueryClient.num_queries[dest_id] = 1				


	def estimate_query_resp_times(self):
		print('Estimating query resp time from {}...'.format(self.city['name']))
		for query in self.queries:
			query.estimate_resp_time(QueryClient.num_queries[query.dest_id])
			
					 
	def __init__(self, conf, dc, topo, city):
		self.conf = conf
		self.dc = dc
		self.topo = topo
		self.city = city
		self.city_radius = math.sqrt(city.land_area / math.pi)
		self.num_queries_per_min = int(city.population * self.conf['ratio_of_adults'] * \
									   self.conf['ratio_of_adult_smartphone_owners'] * \
									   self.conf['num_queries_per_smartphone_per_hour'] / 60)
		self.create_queries()
