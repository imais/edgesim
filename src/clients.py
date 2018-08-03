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
	def set_params(dc, sigma1, sigma2, bw_mobile, bw_wan, bw_lan):
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
		time = Comm.sigma1 + Comm.sigma2 * dist_mi + data / bw
		return time

		
	@staticmethod
	def estimate_query_comm_time(client_lat, client_lng, dc_id, data):
		dc = Comm.dc.loc[dc_id]
		dist_mi = LatLngUtil.compute_dist_mi(client_lat, client_lng,
											 dc.lat, dc.lng)
		time = Comm.sigma1 + Comm.sigma2 * dist_mi + data / Comm.bw_mobile
		return time


class Exec:
	@staticmethod
	def set_params(dc, beta1, beta2, gamma1, gamma2, theta1, theta2, latency):
		Exec.dc = dc
		Exec.beta1 = beta1
		Exec.beta2 = beta2
		Exec.gamma1 = gamma1
		Exec.gamma2 = gamma2
		Exec.theta1 = theta1
		Exec.theta2 = theta2
		Exec.latency = latency

		
	@staticmethod
	def estimate_map_reduce_time(dc_id, data):
		m = Exec.dc.loc[dc_id].m
		return data / (Exec.beta1 + Exec.beta2 * m)

	
	@staticmethod
	def estimate_reduce_time(dc_id, data):
		m = Exec.dc.loc[dc_id].m
		return data / (Exec.gamma1 + Exec.gamma2 * m)

	
	@staticmethod
	def estimate_query_time(dc_id, queries_per_sec):
		m = Exec.dc.loc[dc_id].m
		return Exec.latency * (queries_per_sec / (Exec.theta1 + Exec.theta2 * m))


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
	query_req_bytes = 0
	query_resp_bytes = 0

	@staticmethod
	def set_params(query_req_bytes, query_resp_bytes):
		Query.query_req_bytes = query_req_bytes
		Query.query_resp_bytes = query_resp_bytes	
		
	
	def __init__(self, src_id, src_lat, src_lng, dest_id):
		self.src_id = src_id
		self.src_lat = src_lat
		self.src_lng = src_lng
		self.dest_id = dest_id

	def query_to_str(self, dc, topo):
		s = 'city=' + dc.loc[self.src_id, 'name'] + \
			', (lat, lng)=('  + str(self.src_lat) + ', ' + str(self.src_lng) + '), ' + \
			'dest=' + dc.loc[self.dest_id, 'name'] + \
			', m=' + str(dc.loc[self.dest_id, 'm']) + \
			', qps=' + str(QueryClient.queries_per_sec[self.dest_id]) + \
			', total_resp time=' + str(self.total_resp_time * 1e3) + \
			'ms (comm1=' + str(self.comm_req_time * 1e3) + \
			', query=' + str(self.query_time * 1e3) + \
			', comm2=' + str(self.comm_resp_time * 1e3) + ')'
		return s


	def estimate_resp_time(self, queries_per_sec):
		self.comm_req_time = Comm.estimate_query_comm_time(self.src_lat, self.src_lng, self.dest_id, Query.query_req_bytes)
		if math.isnan(self.comm_req_time):
			raise ValueError('self.comm_req_time is nan')
		
		self.query_time = Exec.estimate_query_time(self.dest_id, queries_per_sec)
		if math.isnan(self.query_time):
			raise ValueError('self.query_time is nan')
		
		self.comm_resp_time = Comm.estimate_query_comm_time(self.src_lat, self.src_lng, self.dest_id, Query.query_resp_bytes)
		if math.isnan(self.comm_resp_time):
			raise ValueError('self.query_time): is nan')		

		self.total_resp_time = self.comm_req_time + self.query_time + self.comm_resp_time


class Hierarchy:
	levels = [['borough', 'cdp', 'city', 'municipality', 'other', 'town', 'township', 'village'], ['county'], ['state'], ['region']]
	L = len(levels)
		

class QueryClient(object):
	queries = {}
	queries_per_sec = {}
	SQ_METER_TO_SQ_MI = 1.0 / (1e6 * (LatLngUtil.KM_PER_MILE**2))
	

	def create_queries(self):
		self.queries = []

		city_others_indices = self.dc.loc[self.dc.type.isin(Hierarchy.levels[0]) & \
										  (self.dc.parent_id == self.city.parent_id) & \
										  (self.dc.index != self.city.name)].index

		county = self.dc.loc[self.city.parent_id]
		county_others_indices = self.dc.loc[self.dc.type.isin(Hierarchy.levels[1]) & \
											(self.dc.parent_id == county.parent_id) & \
											(self.dc.index != county.name)].index
		state =  self.dc.loc[county.parent_id]
		state_others_indices = self.dc.loc[self.dc.type.isin(Hierarchy.levels[2]) & \
										   (self.dc.parent_id == state.parent_id) & \
										   (self.dc.index != state.name)].index

		test_queries = int(self.queries_per_min * self.conf['test_duration_sec'] / 60)
		for i in range(test_queries):
			r = random.uniform(0, 1)
			dest_type = QueryDestType.CITY_SELF			
			total = self.conf['query_dist'][int(dest_type)]
			# query_dist: [city_self, city_others, county_self, county_others, state_self, state_others, region]
			while total < r:
				dest_type += 1
				total += self.conf['query_dist'][int(dest_type)]

			try:
				dest_id = -1
				if dest_type == QueryDestType.CITY_SELF:
					dest_id = self.city.name
				elif 0 < len(city_others_indices) and dest_type == QueryDestType.CITY_OTHER:
					dest_id = random.choice(city_others_indices)
				elif dest_type == QueryDestType.COUNTY_SELF:
					dest_id = county.name
				elif 0 < len(county_others_indices) and dest_type == QueryDestType.COUNTY_OTHER:
					dest_id = random.choice(county_others_indices)
				elif dest_type == QueryDestType.STATE_SELF:
					dest_id = state.name
				elif 0 < len(state_others_indices) and dest_type == QueryDestType.STATE_OTHER:
					dest_id = random.choice(state_others_indices)
				else:
					# dest_type == QueryDestType.REGION
					dest_id = self.dc.loc[state.parent_id].name
			except IndexError:
				print("dest_type={}, city={}".format(dest_type, self.city))
				raise

			(src_lat, src_lng) = LatLngUtil.get_rand_lat_lng(self.city.lat, self.city.lng,
															 360, self.city_radius_mi)
			self.queries.append(Query(self.city.name, src_lat, src_lng, dest_id))
			
			if dest_id in QueryClient.queries:
				QueryClient.queries[dest_id] += 1
			else:
				QueryClient.queries[dest_id] = 1

		for key, value in QueryClient.queries.iteritems():
			QueryClient.queries_per_sec[key] = value / self.conf['test_duration_sec']


	def estimate_query_resp_times(self):
		# print('Estimating query resp time from {}...'.format(self.city['name']))
		for query in self.queries:
			query.estimate_resp_time(QueryClient.queries_per_sec[query.dest_id])
		return self.queries
			
					 
	def __init__(self, conf, dc, topo, city):
		self.conf = conf
		self.dc = dc
		self.topo = topo
		self.city = city
		self.city_radius_mi = math.sqrt(city.land_area * QueryClient.SQ_METER_TO_SQ_MI / math.pi)
		self.queries_per_min = int(city.population * self.conf['ratio_of_adults'] * \
									   self.conf['ratio_of_adult_smartphone_owners'] * \
									   self.conf['queries_per_smartphone_per_hour'] / 60)
		self.create_queries()


class DataAggregationResult:
	def __init__(self, level, aggr_time, exec_time, comm_time,
				 dc1_id, dc1_name, dc1_data_in, dc1_m,
				 dc2_id, dc2_name):
		self.level = level
		self.aggr_time = aggr_time
		self.exec_time = exec_time
		self.comm_time = comm_time
		self.dc1_id = dc1_id
		self.dc1_name = dc1_name
		self.dc1_data_in = dc1_data_in
		self.dc1_m = dc1_m
		self.dc2_id = dc2_id
		self.dc2_name = dc2_name


	def __str__(self):
		s = 'L{}: time(total:{:.3f}, exec:{:.3f}, comm:{:.3f}) ms, dc1(id:{}, name:{}, data_in:{:.3f} bytes, m:{}), dc2(id:{}, name:{})'.format(self.level, self.aggr_time, self.exec_time, self.comm_time, self.dc1_id, self.dc1_name, self.dc1_data_in, self.dc1_m, self.dc2_id, self.dc2_name)
		return s

	
	def to_csv(self):
		s = '{}, {:.3f}, {:.3f}, {:.3f}, {}, {}, {:.3f} {}, {}, {}'.format(self.level, self.aggr_time, self.exec_time, self.comm_time, self.dc1_id, self.dc1_name, self.dc1_data_in, self.dc1_m, self.dc2_id, self.dc2_name)
		return s

		
class DataAggregator:
	mobile_data = 0.0
	wan_data = 0.0
	lan_data = 0.0
	
	@staticmethod
	def set_params(conf, dc, topo):
		DataAggregator.conf = conf
		DataAggregator.dc = dc
		DataAggregator.topo = topo

	@staticmethod
	def aggregate():
		conf = DataAggregator.conf
		dc = DataAggregator.dc
		topo = DataAggregator.topo
		data_out = conf['bytes_reduce_out']

		max_results = []
		for l in range(Hierarchy.L):
			results = []
			entities = topo.loc[topo.type.isin(Hierarchy.levels[l])]
			for index, entity in entities.iterrows():
				if l == 0:
					topo.loc[index, 'data_in'] = dc.loc[index].population * conf['sensors_per_person'] * conf['bytes_per_sensor_per_time_window']
					
				dc1_id = entity.dc_id					
				if l < Hierarchy.L - 1:
					topo.loc[entity.parent_id, 'data_in'] += data_out
					dc2_id = topo.loc[entity.parent_id].dc_id
					exec_time = Exec.estimate_map_reduce_time(dc1_id, topo.loc[index, 'data_in'])
					comm_time = Comm.estimate_dc_comm_time(dc1_id, dc2_id, data_out)
					aggr_time = exec_time + comm_time
					if l == 0:
						DataAggregator.mobile_data += topo.loc[index, 'data_in']
					elif dc1_id == dc2_id:
						DataAggregator.lan_data += topo.loc[index, 'data_in']
					else:
						DataAggregator.wan_data += topo.loc[index, 'data_in']				
				else:
					dc2_id = None
					exec_time = Exec.estimate_map_reduce_time(dc1_id, topo.loc[index, 'data_in'])
					comm_time = 0.0
					aggr_time = exec_time

				# save time results in msec
				result = DataAggregationResult(l, aggr_time*1000, exec_time*1000,
											   comm_time*1000 if comm_time is not None else None,
											   dc1_id, dc.loc[dc1_id, 'name'],
											   topo.loc[index, 'data_in'], dc.loc[dc1_id, 'm'], 
											   dc2_id,
											   dc.loc[dc2_id, 'name'] if dc2_id is not None else None)
				results.append(result)

			max_results.append(max(results, key=lambda (result): result.aggr_time))

		return max_results

	
	@staticmethod
	def get_tx_stats_kb():
		return DataAggregator.mobile_data/1024, DataAggregator.lan_data/1024, DataAggregator.wan_data/1024		
