import logging
from models import Exec, Comm

log = logging.getLogger()


class DataAggregationResult(object):
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
		s = 'L{}: time(total:{:.3f}, exec:{:.3f}, comm:{:.3f}) s, dc1(id:{}, name:{}, data_in:{:.3f} bytes, m:{}), dc2(id:{}, name:{})'.format(self.level, self.aggr_time, self.exec_time, self.comm_time, self.dc1_id, self.dc1_name, self.dc1_data_in, self.dc1_m, self.dc2_id, self.dc2_name)
		return s

	
	def to_csv(self):
		s = '{}, {:.3f}, {:.3f}, {:.3f}, {}, {}, {:.3f} {}, {}, {}'.format(self.level, self.aggr_time, self.exec_time, self.comm_time, self.dc1_id, self.dc1_name, self.dc1_data_in, self.dc1_m, self.dc2_id, self.dc2_name)
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
			for index, entity in entities.iterrows():
				dc1_id = entity.dc_id
				data_in = entity.data_in
				if l < L - 1:
					data_out = entity['data_in'] * conf['alpha'] if l == 0 else 1.0
					topo.loc[entity.parent_id, 'data_in'] += data_out
					dc2_id = topo.loc[entity.parent_id].dc_id
					exec_time = Exec.estimate_map_reduce_time(dc1_id, data_in)
					comm_time = Comm.estimate_dc_comm_time(dc1_id, dc2_id, data_out)
					aggr_time = exec_time + comm_time
					if l == 0:
						DataAggregator.mobile_data += data_in
					elif dc1_id == dc2_id:
						DataAggregator.lan_data += data_in
					else:
						DataAggregator.wan_data += data_in
				else:
					# region DC
					dc2_id = None
					exec_time = Exec.estimate_map_reduce_time(dc1_id, data_in)
					comm_time = 0.0
					aggr_time = exec_time

				# save time results in msec
				result = DataAggregationResult(l, aggr_time, exec_time,
											   comm_time if comm_time is not None else None,
											   dc1_id, dc.loc[dc1_id, 'name'],
											   data_in, dc.loc[dc1_id, 'm'], 
											   dc2_id,
											   dc.loc[dc2_id, 'name'] if dc2_id is not None else None)
				results.append(result)

			max_results.append(max(results, key=lambda (result): result.aggr_time))

		return max_results

	
	@staticmethod
	def get_tx_stats_kb():
		return DataAggregator.mobile_data/1024, DataAggregator.lan_data/1024, DataAggregator.wan_data/1024		
