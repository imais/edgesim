import argparse
import json
import logging
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from clients import Comm, Exec, Query, QueryClient, DataAggregator, DataAggregationResult, Hierarchy

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

	args = parser.parse_args()
	
	return args


def init_conf(args):
	if os.path.exists(args.conf):
		with open(args.conf, 'rt') as f:
			# args have priority over settings in conf
			conf = dict(json.load(f).items() + vars(args).items())
	else:
		conf = args
	return conf


def load_data(mapping, test=False):
	dc_file = DC_FILE_TEST if test else DC_FILE_PROD
	dc = pd.read_csv(dc_file, header=0, index_col=0, sep='\t')
	topo = pd.DataFrame(columns=['type', 'name', 'parent_id', 'dc_id', 'data_in'])
	topo.type = dc.type
	topo.name = dc.name
	topo.parent_id = dc.parent_id
	topo.data_in = 0.0
	if mapping == 'a':
		# Use all level DCs
		topo.dc_id = dc.index
	elif mapping == 'b':
		# Use state and region DCs
		prev_state = None
		for i, row in topo.iterrows():
			if dc.loc[i].state_code != prev_state:
				print('Loading data for {}...'.format(dc.loc[i].state_code))
				prev_state = dc.loc[i].state_code
			if row.type in Hierarchy.levels[0]: # city, town, village, ...
				topo['dc_id'][i] = dc.loc[dc.loc[i].parent_id].parent_id
			elif row.type in Hierarchy.levels[1]:
				topo['dc_id'][i] = dc.loc[i].parent_id
			else:
				# state or region
				topo['dc_id'][i] = i
	elif mapping == 'c':
		# Use region DCs only: there must be only one region DC
		topo.dc_id = dc.loc[dc.type == 'region'].index.values[0]
	return dc, topo


def init(args):
	conf = init_conf(args)
	dc, topo = load_data(conf['mapping'], conf['test'])

	Comm.set_params(dc, conf['sigmas'][0], conf['sigmas'][1], 
					conf['omegas'][0], conf['omegas'][1], conf['omegas'][2]);
	Exec.set_params(dc, conf['betas'][0], conf['betas'][1], 
					conf['gammas'][0], conf['gammas'][1],
					conf['thetas'][0], conf['thetas'][1], conf['lambda'])
	Query.set_params(conf['query_req_bytes'], conf['query_resp_bytes'])
	DataAggregator.set_params(conf, dc, topo)

	return conf, dc, topo
	

def aggregate_data(conf):
	print
	print('### Aggregate Data ###')
	
	results = DataAggregator.aggregate()
	tx_data_stats = DataAggregator.get_tx_stats_kb()	
	total_aggr_time = '{:.3f}'.format(sum(result.aggr_time for result in results))

	if conf['verbose']:
		print("Total Aggregation Time: {} ms".format(total_aggr_time))	
		for result in results:
			print result
		print("Tx Data (mobile/LAN/WAN) = {} Kbytes".format(tx_data_stats))	
	else:
		l = [total_aggr_time]
		l += [result.to_csv() for result in results]
		l += ['{:.3f}, {:.3f}, {:.3f}'.format(tx_data_stats[0], tx_data_stats[1], tx_data_stats[2])]
		print('DataAggrResults: {}'.format(', '.join(l)))


def query_data(conf):
	print
	print('### Query Data ###')
	
	city_ids = dc.loc[dc.type.isin(Hierarchy.levels[0])].index
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
	resp_times = [q.resp_time * 1000 for q in query_results]
	avg_time = '{:.3f}'.format(np.mean(resp_times))
	
	if conf['verbose']:
		print("Response Time:")
		print("Max: {}".format(max_query))
		print("Min: {}".format(min_query))
		print("Avg: {:.3f} ms".format(avg_time))
		# CDF
		num_bins = 40
		counts, bin_edges = np.histogram(resp_times, bins=num_bins, normed=True)
		cdf = np.cumsum(counts)
		plt.plot (bin_edges[1:], cdf/cdf[-1])
		print bin_edges[1:]
		print cdf/cdf[-1]
		plt.show()
	else:
		l = [max_query.to_csv(), min_query.to_csv(), avg_time]
		print('QueryResults: {}'.format(', '.join(l)))
		

def main(args):
	global dc, topo, resp_times
	conf, dc, topo = init(args)
	log.info('Configs: {}'.format(conf))
	aggregate_data(conf)
	query_data(conf)
	
	
if __name__ == '__main__':
	args = parse_args()
	main(args)
