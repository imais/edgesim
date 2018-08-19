import logging
import pandas as pd
from models import Exec, Comm

log = logging.getLogger()


class DataAggregationResult(object):
	def __init__(self, level, aggr_time, exec_time, comm_time,
				 topo1_id, topo1_name, topo2_id, topo2_name, dist_mi,
				 dc1_id, dc1_name, dc1_data_in, dc1_m,
				 dc2_id, dc2_name):
		self.level = level
		self.aggr_time = aggr_time
		self.exec_time = exec_time
		self.comm_time = comm_time
		self.topo1_id = topo1_id
		self.topo1_name = topo1_name
		self.topo2_id = topo2_id
		self.topo2_name = topo2_name
		self.dist_mi = dist_mi
		self.dc1_id = dc1_id
		self.dc1_name = dc1_name
		self.dc1_data_in = dc1_data_in
		self.dc1_m = dc1_m
		self.dc2_id = dc2_id
		self.dc2_name = dc2_name


	def __str__(self):
		s = 'L{}: time(total:{:.3f}, exec:{:.3f}, comm:{:.3f}) s, topo1(id:{}, name:{}), topo2(id:{}, name:{}), dist_mi: {:.3f}, dc1(id:{}, name:{}, data_in:{:.3f} mb, m:{}), dc2(id:{}, name:{})'.format(self.level, self.aggr_time, self.exec_time, self.comm_time, self.topo1_id, self.topo1_name, self.topo2_id, self.topo2_name, self.dist_mi, self.dc1_id, self.dc1_name, (self.dc1_data_in/1e6), self.dc1_m, self.dc2_id, self.dc2_name)
		return s

	
	def to_csv(self):
		s = '{}, {:.3f}, {:.3f}, {:.3f}, {}, {}, {}, {}, {}, {}, {:.3f}, {}, {}, {}'.format(self.level, self.aggr_time, self.exec_time, self.comm_time, self.topo1_id, self.topo1_name, self.topo2_id, self.topo2_name, self.dc1_id, self.dc1_name, (self.dc1_data_in/1e6), self.dc1_m, self.dc2_id, self.dc2_name)		
		return s

		
class DataAggregator(object):
	mobile_data = 0.0
	wan_data = 0.0
	lan_data = 0.0
	
	@staticmethod
	def set_params(conf, dc, topo):
		DataAggregator.conf = conf
		DataAggregator.dc = dc
		DataAggregator.topo = topo

	@staticmethod
	def aggregate(levels):
		# levels: [city_aliases, ['county'], ['state'], ['region']]
		L = len(levels)
		
		conf = DataAggregator.conf
		dc = DataAggregator.dc
		topo = DataAggregator.topo

		max_results = []
		for l in range(L):
			results = []
			entities = topo.loc[topo.type.isin(levels[str(l)])]
			max_usage = pd.DataFrame(data=0.0, index=dc.index, columns=['max_usage'])

			# collect incoming data per dc per level for map-reduce exec time
			dc.loc[topo.loc[entities.index, 'dc_id'], 'data_in'] = 0.0
			for index, entity in entities.iterrows():
				dc.loc[entity.dc_id, 'data_in'] += entity.data_in
			
			for index, entity in entities.iterrows():
				dc1_id = entity.dc_id
				dc_data_in = dc.loc[dc1_id, 'data_in']

				m = 0
				if l < L - 1:
					# map-reduce: compute based on data_in per dc
					exec_time = 0.0					
					if l == 0:
						exec_time += Exec.estimate_map_time(dc1_id, dc_data_in)
						dc_data_in *= conf['s_m']
					exec_time += Exec.estimate_reduce_time(dc1_id, dc_data_in)
					
					# communcation: compute based on data_in per entity  
					data_out = entity['data_in'] * (conf['s_m'] if l == 0 else 1.0) * conf['s_r']
					topo.loc[entity.parent_id, 'data_in'] += data_out
					dc2_id = topo.loc[entity.parent_id].dc_id

					comm_time, dist_mi = Comm.estimate_dc_comm_time(dc1_id, dc2_id, data_out)
					aggr_time = exec_time + comm_time

					# stats
					if l == 0:
						# incoming data to level 0 are all mobile
						DataAggregator.mobile_data += entity['data_in']
					if dc1_id == dc2_id:
						DataAggregator.lan_data += data_out
					else:
						DataAggregator.wan_data += data_out

				else:
					# region DC
					dc2_id = None
					exec_time = Exec.estimate_reduce_time(dc1_id, dc_data_in)
					comm_time = 0.0
					dist_mi = 0
					aggr_time = exec_time

				# keep track of max usage of machines per dc for level l
				max_usage.loc[dc1_id, 'max_usage'] = max(max_usage.loc[dc1_id, 'max_usage'], aggr_time)
					
				# save time results in msec
				result = DataAggregationResult(l, aggr_time, exec_time,
											   comm_time if comm_time is not None else None,
											   entity.name, entity['name'],
											   entity.parent_id, topo.loc[entity.parent_id, 'name'] if entity.parent_id != -1 else None,
											   dist_mi,
											   dc1_id, dc.loc[dc1_id, 'name'],
											   dc_data_in, dc.loc[dc1_id, 'm'], 
											   dc2_id,
											   dc.loc[dc2_id, 'name'] if dc2_id is not None else None)
				results.append(result)

			dc['max_usage'] += max_usage['max_usage']
			max_results.append(max(results, key=lambda (result): result.aggr_time))

		return max_results

	
	@staticmethod
	def get_tx_stats(unit):
		return DataAggregator.mobile_data/unit, DataAggregator.wan_data/unit, DataAggregator.lan_data/unit

	@staticmethod
	def get_machine_hours():
		machine_hours = sum(DataAggregator.dc.apply(lambda x: x['m'] * x['max_usage'] / 3600, axis=1))
		return machine_hours
