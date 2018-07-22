import argparse
import json
import logging
import logging.config
import pandas as pd
import os
from clients import QueryClient
# from node import NodeType, Node


log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# constants
ALPHAS =  {'city': 0.5, 'county': 0.5, 'region': 0.5}
MACHINES = {'city': 1, 'county': 5, 'region': 10}
PRODUCER_DATA_SIZE = 1024  # bytes
PRODUCER_AVG_INTERVAL = 2 # seconds
PRODUCER_TIME_WINDOW = 10 # seconds
GEO_DB_FILE = "./data/test-db.csv"


# globals
geo_db = pd.DataFrame.from_csv(GEO_DB_FILE, header=0)
conf = None

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-cf',	'--conf', default='./conf/conf.json', type=str)	
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


def init_query_clients(conf):
	city_ids = geo_db.loc[geo_db.type == 'city'].index
	query_clients = [QueryClient(conf, geo_db, src_id) for src_id in city_ids]
	return query_clients
	

def init(args):
	conf = init_conf(args)
	query_clients = init_query_clients(conf)
	return conf
	

def main(args):
	conf = init(args)
	log.info('Configs: {}'.format(conf))

	
if __name__ == '__main__':
	args = parse_args()
	main(args)
