import logging
import numpy as np
import random
from enum import Enum

log = logging.getLogger()

# constants
# https://www.techwalla.com/articles/network-latency-milliseconds-per-mile
LATENCY_SEC_PER_MILE = 0.00001  # 0.01 msec
THROUGHPUT_PER_MACHINE = 



class NodeType(Enum):
	NULL = 0
	CITY = 1
	COUNTY = 2
	REGION = 3

	def __str__(self):
		return self.name


class Node(object):
	def __init__(self, id, node_type, x, y, m, alpha):
		self.id = id
		self.node_type = node_type
		self.x = x
		self.y = y
		self.m = m
		self.input_data = 0.
		self.output_data = 0.
		self.alpha = alpha # d(out) = alpha * d(in)

	def set_parent(self, parent):
		self.parent = parent
				
	def set_input_data(self, data):
		self.input_data = data

	def compute(self):
		self.output_data = self.input_data * self.alpha




