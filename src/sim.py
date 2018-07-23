import argparse
import json
import logging
import logging.config
import pandas as pd
import os
from clients import Comm, Exec, QueryClient

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

DC_FILE = "./data/test-dc.tsv"


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
	topo = pd.DataFrame(columns=['type', 'name', 'parent_id', 'dc_id'])
	topo.type = dc.type
	topo.name = dc.name
	topo.parent_id = dc.parent_id
	if mapping == 'a':
		# Use all level DCs
		topo.dc_id = dc.index
	elif mapping == 'b':
		# Use state and region DCs
		for i, row in topo.iterrows():
			if row.type == 'city':
				topo['dc_id'][i] = dc.loc[dc.loc[i].parent_id].parent_id
			elif row.type == 'county':
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

	Comm.init(dc, conf['sigmas'][0], conf['sigmas'][1], 
			  conf['omegas'][0], conf['omegas'][1], conf['omegas'][2]);
	Exec.init(dc, conf['betas'][0], conf['betas'][1], 
			  conf['gammas'][0], conf['gammas'][1],
			  conf['thetas'][0], conf['thetas'][1])

	city_ids = dc.loc[dc.type == 'city'].index
	query_clients = []
	for city_id in city_ids:
		print('Creating query client for {}...'.format(dc.loc[city_id]['name']))
		query_clients.append(QueryClient(conf, dc, topo, dc.loc[city_id]))

	return conf, dc, topo, query_clients
	

def main(args):
	global dc, topo
	conf, dc, topo, query_clients = init(args)
	log.info('Configs: {}'.format(conf))

	for query_client in query_clients:
		query_client.estimate_query_resp_times()

	
if __name__ == '__main__':
	args = parse_args()
	main(args)
