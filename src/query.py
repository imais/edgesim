import logging
import math
import random
from latlng_util import LatLngUtil
from enum import IntEnum
from models import Exec, Comm


log = logging.getLogger()


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
		
	
	def __init__(self, src_id, src_lat, src_lng, dest_id, dc_id):
		self.src_id = src_id
		self.src_lat = src_lat
		self.src_lng = src_lng
		self.dest_id = dest_id
		self.dc_id = dc_id		


	def estimate_resp_time(self, queries_per_sec):
		self.comm_req_time, _ = Comm.estimate_query_comm_time(self.src_lat, self.src_lng, self.dc_id, Query.query_req_bytes)
		if math.isnan(self.comm_req_time):
			raise ValueError('self.comm_req_time is nan')
		
		self.query_time = Exec.estimate_query_time(self.dc_id, queries_per_sec)
		if math.isnan(self.query_time):
			raise ValueError('self.quepry_time is nan')
		
		self.comm_resp_time, self.dist_mi = Comm.estimate_query_comm_time(self.src_lat, self.src_lng, self.dc_id, Query.query_resp_bytes)
		if math.isnan(self.comm_resp_time):
			raise ValueError('self.query_time): is nan')		

		self.total_resp_time = self.comm_req_time + self.query_time + self.comm_resp_time


class QueryResult(object):
	def __init__(self, resp_time, comm_req_time, query_time, comm_resp_time, dist_mi,
				 src_id, src_name, src_lat, src_lng, 
				 dest_id, dest_name, dest_type,
				 dc_id, dc_name, dc_type, dc_qps, dc_m):
		self.resp_time = resp_time
		self.comm_req_time = comm_req_time
		self.query_time = query_time
		self.comm_resp_time = comm_resp_time
		self.dist_mi = dist_mi
		self.src_id = src_id
		self.src_name = src_name
		self.src_lat = src_lat
		self.src_lng = src_lng
		self.dest_id = dest_id
		self.dest_type = dest_type
		self.dest_name = dest_name
		self.dc_id = dc_id
		self.dc_type = dc_type
		self.dc_name = dc_name		
		self.dc_qps = dc_qps
		self.dc_m = dc_m
		

	def __str__(self):
		s = 'time(total:{:.5f}, comm_req:{:.5f}, query:{:.5f}, comm_resp:{:.5f}) ms, {:.5f} mi, src(id:{}, name:{}, lat:{}, lng:{}), dest(id:{}, type:{}, name:{}), dc(id:{}, type:{}, name:{}, qps:{:.5f}, m:{})'.format(self.resp_time, self.comm_req_time, self.query_time, self.comm_resp_time, self.dist_mi, self.src_id, self.src_name, self.src_lat, self.src_lng, self.dest_id, self.dest_type, self.dest_name, self.dc_id, self.dc_type, self.dc_name, self.dc_qps, self.dc_m)
		return s


	def to_csv(self):
		s = '{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:.5f}, {}'.format(self.resp_time, self.comm_req_time, self.query_time, self.comm_resp_time, self.dist_mi, self.src_id, self.src_name, self.src_lat, self.src_lng, self.dest_id, self.dest_type, self.dest_name, self.dc_id, self.dc_type, self.dc_name, self.dc_qps, self.dc_m)		
		return s
		
	

class QueryClient(object):
	queries = {}
	queries_per_sec = {}
	SQ_METER_TO_SQ_MI = 1.0 / (1e6 * (LatLngUtil.KM_PER_MILE**2))
	

	def create_queries(self, levels):
		self.queries = []

		city_ids = self.topo.loc[self.topo.type.isin(levels['0']) & \
								 (self.topo.parent_id == self.city.parent_id) & \
								 (self.topo.index != self.city.name)].index
		county = self.topo.loc[self.city.parent_id]
		county_ids = self.topo.loc[self.topo.type.isin(levels['1']) & \
								   (self.topo.parent_id == county.parent_id) & \
								   (self.topo.index != county.name)].index
		state =  self.topo.loc[county.parent_id]
		state_ids = self.topo.loc[self.topo.type.isin(levels['2']) & \
								  (self.topo.parent_id == state.parent_id) & \
								  (self.topo.index != state.name)].index

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
				elif 0 < len(city_ids) and dest_type == QueryDestType.CITY_OTHER:
					dest_id = random.choice(city_ids)
				elif dest_type == QueryDestType.COUNTY_SELF:
					dest_id = county.name
				elif 0 < len(county_ids) and dest_type == QueryDestType.COUNTY_OTHER:
					dest_id = random.choice(county_ids)
				elif dest_type == QueryDestType.STATE_SELF:
					dest_id = state.name
				elif 0 < len(state_ids) and dest_type == QueryDestType.STATE_OTHER:
					dest_id = random.choice(state_ids)
				else:
					# dest_type == QueryDestType.REGION
					dest_id = self.topo.loc[state.parent_id].name
			except IndexError:
				print("dest_type={}, city={}".format(dest_type, self.city))
				raise

			dc_id = self.topo.loc[dest_id].dc_id

			(src_lat, src_lng) = LatLngUtil.get_rand_lat_lng(self.city.lat, self.city.lng,
															 360, self.city_radius_mi)
			self.queries.append(Query(self.city.name, src_lat, src_lng, dest_id, dc_id))
			
			if dc_id in QueryClient.queries:
				QueryClient.queries[dc_id] += 1
			else:
				QueryClient.queries[dc_id] = 1

		for key, value in QueryClient.queries.iteritems():
			QueryClient.queries_per_sec[key] = value / self.conf['test_duration_sec']


	def query(self):
		# print('Estimating query resp time from {}...'.format(self.city['name']))
		results = []
		for query in self.queries:
			query.estimate_resp_time(QueryClient.queries_per_sec[query.dc_id])
			# save time in msec
			result = QueryResult(query.total_resp_time*1000, query.comm_req_time*1000,
								 query.query_time*1000, query.comm_resp_time*1000,
								 query.dist_mi,
								 query.src_id, self.topo.loc[query.src_id, 'name'],
								 query.src_lat, query.src_lng,
								 query.dest_id, self.topo.loc[query.dest_id, 'name'],
								 self.topo.loc[query.dest_id, 'type'],
								 query.dc_id, self.dc.loc[query.dc_id, 'name'],
								 self.dc.loc[query.dc_id, 'type'],
								 QueryClient.queries_per_sec[query.dc_id],
								 self.dc.loc[query.dc_id, 'm'])
			results.append(result)
		return results
			
					 
	def __init__(self, conf, dc, topo, city):
		self.conf = conf
		self.dc = dc
		self.topo = topo
		self.city = city
		self.city_radius_mi = math.sqrt(city.land_area * QueryClient.SQ_METER_TO_SQ_MI / math.pi)
		self.queries_per_min = int(city.population * self.conf['ratio_of_adults'] * \
									   self.conf['ratio_of_adult_smartphone_owners'] * \
									   self.conf['queries_per_smartphone_per_hour'] / 60)
		self.create_queries(self.conf['levels'])
