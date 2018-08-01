import argparse
import json
import logging
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from clients import Comm, Exec, Query, QueryClient, DataAggregator, Hierarchy

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# DC_FILE = "./data/test-dc.tsv"
DC_FILE = "./data/us-northeast.tsv"


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--conf', default='./conf/conf.json', type=str)
	parser.add_argument('-m', '--mapping', default='a', choices=['a', 'b', 'c'], type=str)		
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


def load_data(mapping):
	dc = pd.read_csv(DC_FILE, header=0, index_col=0, sep='\t')
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
	dc, topo = load_data(conf['mapping'])

	Comm.set_params(dc, conf['sigmas'][0], conf['sigmas'][1], 
					conf['omegas'][0], conf['omegas'][1], conf['omegas'][2]);
	Exec.set_params(dc, conf['betas'][0], conf['betas'][1], 
					conf['gammas'][0], conf['gammas'][1],
					conf['thetas'][0], conf['thetas'][1], conf['lambda'])
	Query.set_params(conf['query_req_bytes'], conf['query_resp_bytes'])
	DataAggregator.set_params(conf, dc, topo)

	return conf, dc, topo
	

def main(args):
	global dc, topo, resp_times
	conf, dc, topo = init(args)
	log.info('Configs: {}'.format(conf))

	# Estimate data aggregation time
	aggr_results = DataAggregator.aggregate()	

	# Estimate query response times
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
		query_results += query_client.estimate_query_resp_times()

	# Display results
	print
	total_aggr_time = sum(result[0] for result in aggr_results)
	print("Total Aggregation Time: {}ms".format(total_aggr_time))
	for l in range(Hierarchy.L):
		print("L{}: {}".format(l+1, DataAggregator.result_to_str(aggr_results[l])))

	print
	print("Response Time:")
	max_query = max(query_results, key=lambda q: q.total_resp_time)
	min_query = min(query_results, key=lambda q: q.total_resp_time)
	print("Max: {}".format(max_query.query_to_str(dc, topo)))
	print("Min: {}".format(min_query.query_to_str(dc, topo)))
	resp_times_ms = [q.total_resp_time * 1000 for q in query_results]	
	print("Avg: {}ms".format(np.mean(resp_times_ms)))

	num_bins = 40
	counts, bin_edges = np.histogram(resp_times_ms, bins=num_bins, normed=True)
	cdf = np.cumsum(counts)
	plt.plot (bin_edges[1:], cdf/cdf[-1])
	print bin_edges[1:]
	print cdf/cdf[-1]
	plt.show()		  
		
	
if __name__ == '__main__':
	args = parse_args()
	main(args)
