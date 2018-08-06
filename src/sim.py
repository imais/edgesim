import argparse
import json
import logging
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from data_aggregator import DataAggregator, DataAggregationResult
from query import Query, QueryClient
from models import Comm, Exec

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

DC_FILE_TEST = "./data/test-dc.tsv"
DC_FILE_PROD = "./data/us-northeast.tsv"


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-c', '--conf', default='./conf/conf.json', type=str)
	parser.add_argument('-m', '--mapping', default='a', choices=['a', 'b', 'c'], type=str)
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('-t', '--test', action='store_true')
	parser.add_argument('-d', '--data_aggregation', action='store_true')
	parser.add_argument('-q', '--query', action='store_true')
	parser.add_argument('-a', '--alpha_', type=float)
	parser.add_argument('-s', '--sensors_per_person_', type=float)
	parser.add_argument('-l', '--lambda_ms_', type=float)		

	args = parser.parse_args()
	
	return args


def init_conf(args):
	if os.path.exists(args.conf):
		with open(args.conf, 'rt') as f:
			# args have priority over settings in conf
			conf = dict(json.load(f).items() + vars(args).items())

			# overwrite entries in conf with arguments
			if conf['alpha_'] is not None:
				conf['alpha'] = conf['alpha_']
			del conf['alpha_']
			if conf['sensors_per_person_'] is not None:
				conf['sensors_per_person'] = conf['sensors_per_person_']
			del conf['sensors_per_person_']
			if conf['lambda_ms_'] is not None:
				conf['lambda_ms'] = conf['lambda_ms_']
			del conf['lambda_ms_']
	else:
		conf = args
	return conf


def load_dc(conf):
	# load dc data
	dc_file = DC_FILE_TEST if conf['test'] else DC_FILE_PROD
	dc = pd.read_csv(dc_file, header=0, index_col=0, sep='\t')

	dc['m'] = ''	
	if conf['machine']['alloc_policy'] == 'population':
		L = len(conf['levels'])
		divisor = 1
		population_per_machine = conf['machine']['population']['population_per_machine']
		for l in range(L):
			target_ids = dc.loc[dc.type.isin(conf['levels'][str(l)])].index
			# avg_m = conf['machine']['population']['avg_machines'][l]
			# min_m = conf['machine']['population']['min_machines'][l]
			# total_m = avg_m * len(target_ids)			
			# total_population = sum(dc.loc[target_ids, 'population'])
			# dc['m'] = dc.loc[target_ids].\
			# 		  apply(lambda x: \
			# 				max(min_m, \
			# 					int(total_m * x['population'] / total_population)),
			# 				axis = 1)
			min_m = conf['machine']['population']['min_machines'][l]
			dc.loc[target_ids, 'm'] = dc.loc[target_ids].apply(lambda x: max(min_m, int(x['population'] / population_per_machine / divisor)), axis = 1)
			divisor *= 2
			# machines:
			#   cities   = {mean: 2.01, max: 26, min: 2},
			#   counties = {mean: 4.17, max: 13, min: 4},
			#   states   = {mean: 17.78, [8, 17, 8, 8, 22, 49, 32, 8, 8]},
			#   region   = 70
	elif conf['machine']['alloc_policy'] == 'fixed':
		raise NotImplementedError('fixed allocation policy not implementd')

	return dc


def load_topo(conf, dc):
	# configure topo based on mapping policy
	topo = pd.DataFrame(columns=['type', 'name', 'parent_id', 'dc_id', 'data_in'])
	topo.type = dc.type
	topo.name = dc.name
	topo.parent_id = dc.parent_id
	topo.data_in = 0.0

	# data_in
	city_ids = topo.loc[topo.type.isin(conf['levels'][str(0)])].index
	topo.loc[city_ids, 'data_in'] = topo.loc[city_ids].apply(lambda x: dc.loc[x.name, 'population'] * conf['sensors_per_person'] * conf['bytes_per_sensor_per_time_window'], axis=1)
	
	if conf['mapping'] == 'a':
		# Use all level DCs
		topo.dc_id = dc.index
	elif conf['mapping'] == 'b':
		# Use state and region DCs
		prev_state = None
		for i, row in topo.iterrows():
			if dc.loc[i].state_code != prev_state:
				print('Loading data for {}...'.format(dc.loc[i].state_code))
				prev_state = dc.loc[i].state_code
				
			if row.type in conf['levels'][str(0)]: # city, town, village, ...
				topo['dc_id'][i] = dc.loc[dc.loc[i].parent_id].parent_id
			elif row.type in conf['levels'][str(1)]: # county
				topo['dc_id'][i] = dc.loc[i].parent_id
			else:
				# state or region
				topo['dc_id'][i] = i
	elif conf['mapping'] == 'c':
		# Use region DCs only: there must be only one region DC
		topo.dc_id = dc.loc[dc.type == 'region'].index.values[0]

	return topo


def init(args):
	conf = init_conf(args)
	dc = load_dc(conf)
	topo = load_topo(conf, dc)

	Comm.set_params(dc, conf['sigmas'][0], conf['sigmas'][1], 
					conf['omegas'][0], conf['omegas'][1], conf['omegas'][2]);
	Exec.set_params(dc, conf['betas'][0], conf['betas'][1], 
					conf['gammas'][0], conf['gammas'][1],
					conf['thetas'][0], conf['thetas'][1], conf['lambda_ms']/1000)
	Query.set_params(conf['query_req_bytes'], conf['query_resp_bytes'])
	DataAggregator.set_params(conf, dc, topo)

	return conf, dc, topo
	

def aggregate_data(conf):
	print
	print('### Aggregate Data ###')

	results = DataAggregator.aggregate(conf['levels'])
	tx_data_stats = DataAggregator.get_tx_stats_kb()	
	total_aggr_time = '{:.3f}'.format(sum(result.aggr_time for result in results))

	if conf['verbose']:
		print('Total Aggregation Time: {} s'.format(total_aggr_time))
		print('alpha={}, sensors_per_person={}'\
			  .format(conf['alpha'][0], conf['sensors_per_person']))
		for result in results:
			print result
		print("Tx Data (mobile/LAN/WAN) = {} Kbytes".format(tx_data_stats))	
	else:
		l = [total_aggr_time]
		l += ['{}, {}'.format(conf['alpha'], conf['sensors_per_person'])]
		l += [result.to_csv() for result in results]
		l += ['{:.3f}, {:.3f}, {:.3f}'.format(tx_data_stats[0], tx_data_stats[1], tx_data_stats[2])]
		print('DataAggrResults: {}'.format(', '.join(l)))


def query_data(conf):
	print
	print('### Query Data ###')
	
	city_ids = dc.loc[dc.type.isin(conf['levels']['0'])].index
	query_clients = []
	prev_state = None
	for city_id in city_ids:
		if dc.loc[city_id].state_code != prev_state:
			print('Creating query clients for {}...'.format(dc.loc[city_id].state_code))
			prev_state = dc.loc[city_id].state_code
		query_clients.append(QueryClient(conf, dc, topo, dc.loc[city_id]))	

	query_results = []
	prev_state = None
	for query_client in query_clients:
		if dc.loc[query_client.city.name].state_code != prev_state:
			print('Estimating query resp time for {}...'.format(dc.loc[query_client.city.name].state_code))
			prev_state = dc.loc[query_client.city.name].state_code
		query_results += query_client.query()

	max_query = max(query_results, key=lambda q: q.resp_time)
	min_query = min(query_results, key=lambda q: q.resp_time)
	resp_times = [q.resp_time for q in query_results]
	avg_time = '{:.3f}'.format(np.mean(resp_times))
	
	if conf['verbose']:
		print('Response Time:')
		print('lambda={} ms'.format(conf['lambda_ms']))
		print('Max: {}'.format(max_query))
		print('Min: {}'.format(min_query))
		print('Avg: {} ms'.format(avg_time))
		# CDF
		num_bins = 40
		counts, bin_edges = np.histogram(resp_times, bins=num_bins, normed=True)
		cdf = np.cumsum(counts)
		plt.plot (bin_edges[1:], cdf/cdf[-1])
		print bin_edges[1:]
		print cdf/cdf[-1]
		plt.show()
	else:
		l = ['{}'.format(conf['lambda'])]
		l += [max_query.to_csv(), min_query.to_csv(), avg_time]
		print('QueryResults: {}'.format(', '.join(l)))
		

def main(args):
	global dc, topo, resp_times
	conf, dc, topo = init(args)
	log.info('Configs: {}'.format(conf))
	if not conf['data_aggregation'] and not conf['query']:
		# default: run both
		aggregate_data(conf)
		query_data(conf)
	if conf['data_aggregation']:
		aggregate_data(conf)
	if conf['query']:
		query_data(conf)		
	
if __name__ == '__main__':
	args = parse_args()
	main(args)
