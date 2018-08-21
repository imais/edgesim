from __future__ import print_function
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
	parser.add_argument('-sm','--s_map', type=float)
	parser.add_argument('-sr','--s_reduce', type=float)	
	parser.add_argument('-s', '--sensors_per_person_', type=float)
	parser.add_argument('-l', '--lambda_ms_', type=float)
	parser.add_argument('-rt','--read_topo', action='store_true')
	parser.add_argument('-wt','--write_topo_and_done', action='store_true')
	parser.add_argument('-ap','--alloc_policy', default='population', choices=['population', 'fixed'], type=str)

	args = parser.parse_args()
	
	return args


def init_conf(args):
	if os.path.exists(args.conf):
		with open(args.conf, 'rt') as f:
			# args have priority over settings in conf
			conf = dict(json.load(f).items() + vars(args).items())

			# overwrite entries in conf with arguments
			if conf['alloc_policy'] is not None:
				conf['machine']['alloc_policy'] = conf['alloc_policy']
				del conf['alloc_policy']
				
			if conf['s_map'] is not None:
				conf['s_m'] = conf['s_map']
				del conf['s_map']
				
			if conf['s_reduce'] is not None:
				conf['s_r'] = conf['s_reduce']
				del conf['s_reduce']
				
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
	global dc_file
	print('Loading DC data...')
	
	# load dc data
	dc_file = DC_FILE_TEST if conf['test'] else DC_FILE_PROD
	dc = pd.read_csv(dc_file, header=0, index_col=0, sep='\t')

	dc['m'] = ''	
	L = len(conf['levels'])
	population_per_machine = conf['machine']['population']['population_per_machine']
	for l in range(L):
		ids = dc.loc[dc.type.isin(conf['levels'][str(l)])].index
		if conf['machine']['alloc_policy'] == 'population':
			dc.loc[ids, 'm'] = dc.loc[ids].apply(lambda x: max(1, int((x['population'] / population_per_machine))), axis = 1)
		elif conf['machine']['alloc_policy'] == 'fixed':
			dc.loc[ids, 'm'] = conf['machine']['fixed']	[l]
		else:
			raise NotImplementedError('unfedined allocation policy specified')
		m = dc.loc[ids, 'm']
		print('L{} m: mean={}, max={}, min={}, total={}, n={}'.format(l, np.mean(m), np.max(m), np.min(m), sum(m), len(m)))

	# update later when topology is decided
	dc['data_in'] = 0.0

	# to compute machine hours, in seconds
	dc['max_usage'] = 0.0
	
	return dc


def read_topo(mapping):
	global dc_file
	file_no_extension, _ = os.path.splitext(dc_file)
	file = file_no_extension + '_topo-' + mapping + '.tsv'	
	print('Reading topo file: {}'.format(file))
	topo = pd.read_csv(file, header=0, index_col=0, sep='\t')
	return topo

	
def write_topo(topo, mapping):
	global dc_file
	file_no_extension, _ = os.path.splitext(dc_file)
	file = file_no_extension + '_topo-' + mapping + '.tsv'
	print('Writing topo file: {}'.format(file))	
	topo.to_csv(file, sep='\t')

	
def load_topo(conf, dc):
	print('Creating topology...', end='')

	# configure topo based on mapping policy	
	if conf['read_topo']:
		topo = read_topo(conf['mapping'])
	else:
		topo = pd.DataFrame(columns=['type', 'name', 'parent_id', 'dc_id', 'data_in'])
		topo.type = dc.type
		topo.name = dc.name
		topo.parent_id = dc.parent_id
		topo.data_in = 0.0

		if conf['mapping'] == 'a':
			# Use all level DCs
			topo.dc_id = dc.index
		elif conf['mapping'] == 'b':
			# Use state and region DCs
			prev_state = ''
			for i, row in topo.iterrows():
				if prev_state != dc.loc[i].state_code:
					print('.', end='')
					sys.stdout.flush()
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

	city_ids = topo.loc[topo.type.isin(conf['levels'][str(0)])].index
	topo.loc[city_ids, 'data_in'] = topo.loc[city_ids].apply(lambda x: dc.loc[x.name, 'population'] * conf['sensors_per_person'] * conf['bytes_per_sensor_per_time_window'], axis=1)

	print('')	

	if conf['write_topo_and_done']:
		write_topo(topo, conf['mapping'])

	return topo


def init(args):
	conf = init_conf(args)
	dc = load_dc(conf)

	assert(not conf['read_topo'] or not conf['write_topo_and_done']), "Both read_topo and write_topo_and_done cannot be True!"
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
	print('Simulating data aggregation...')

	results = DataAggregator.aggregate(conf['levels'])
	tx_stats = DataAggregator.get_tx_stats(1e9) # Gbytes
	mh = DataAggregator.get_machine_hours()
	total_aggr_time = sum(result.aggr_time for result in results)
	if conf['machine']['alloc_policy'] == 'population':
		alloc_policy = 'population'
	elif conf['machine']['alloc_policy'] == 'fixed':
		alloc_policy = str(conf['machine']['fixed'])
	else:
		alloc_policy = 'None'

	if conf['verbose']:
		print('Total Aggregation Time: {:.5f} s'.format(total_aggr_time))
		print('mapping={}, s_m={}, s_r={}, sensors_per_person={}, alloc_policy={}'\
			  .format(conf['mapping'], conf['s_m'], conf['s_r'], conf['sensors_per_person'], alloc_policy))
		for result in results:
			print(result)
		print("Tx Data (mobile/WAN/LAN) = {} Gbytes, Machine hours = {}, {}".format(tx_stats, mh, sum(mh)))
	else:
		l = ['{}, {}, {}, {}'.format(conf['mapping'], conf['s_m'], conf['s_r'], alloc_policy)]
		l += ['{:5f}'.format(total_aggr_time)]
		l += [result.to_csv() for result in results]
		l += ['{:.5f}, {:.5f}, {:.5f}'.format(tx_stats[0], tx_stats[1], tx_stats[2])]
		l += ['{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(mh[0], mh[1], mh[2], mh[3], sum(mh))]
		print('DataAggrResults: {}'.format(', '.join(l)))


def query_data(conf):
	global dc, topo
	
	print('Simulating data queries...')	
	
	city_ids = topo.loc[topo.type.isin(conf['levels']['0'])].index
	query_clients = []
	prev_state = None
	print('Creating query clients...')
	for city_id in city_ids:
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
	avg_time = np.mean(resp_times)

	# print('All queries:')
	# for q in query_results:
	# 	print('{}'.format(q))
	
	if conf['verbose']:
		print('Response Time:')
		print('mapping={}, lambda={} ms'.format(conf['mapping'], conf['lambda_ms']))
		print('Max: {}'.format(max_query))
		print('Min: {}'.format(min_query))
		print('Avg: {:.5f} ms'.format(avg_time))
	else:
		l = ['{}, {}'.format(conf['lambda_ms'], conf['mapping'])]
		l += [max_query.to_csv(), min_query.to_csv(), '{:.5f}'.format(avg_time)]
		print('QueryResults: {}'.format(', '.join(l)))

	# CDF
	num_bins = 40
	counts, bin_edges = np.histogram(resp_times, bins=num_bins, normed=True)
	cdf = np.cumsum(counts)
	print('CDF: {}, {}, {}, {}'.format(conf['mapping'], conf['lambda_ms'], bin_edges[1:].tolist(), (cdf/cdf[-1]).tolist()))
	if conf['verbose']:
		plt.plot (bin_edges[1:], cdf/cdf[-1])
		plt.show()		
		

def main(args):
	global dc, topo, resp_times
	conf, dc, topo = init(args)

	if not conf['write_topo_and_done']:
		log.info('Configs: {}'.format(conf))
		if not conf['data_aggregation'] and not conf['query']:
			# default: run both
			aggregate_data(conf)
			query_data(conf)
		if conf['data_aggregation']:
			aggregate_data(conf)
		if conf['query']:
			query_data(conf)

	print('Done!')
	print('')
		
	
if __name__ == '__main__':
	args = parse_args()
	main(args)
