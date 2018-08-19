import logging
import math
from latlng_util import LatLngUtil


log = logging.getLogger()


class Comm:
	@staticmethod
	def set_params(dc, sigma1, sigma2, bw_mobile, bw_wan, bw_lan):
		Comm.dc = dc
		Comm.sigma1 = sigma1
		Comm.sigma2 = sigma2
		Comm.bw_mobile = bw_mobile
		Comm.bw_wan = bw_wan
		Comm.bw_lan = bw_lan
		
		
	@staticmethod
	def estimate_dc_comm_time(dc1_id, dc2_id, data):
		dc1 = Comm.dc.loc[dc1_id]
		dc2 = Comm.dc.loc[dc2_id]		
		dist_mi = LatLngUtil.compute_dist_mi(dc1.lat, dc1.lng,
											 dc2.lat, dc2.lng)
		bw = Comm.bw_lan if dc1_id == dc2_id else Comm.bw_wan
		time = Comm.sigma1 + Comm.sigma2 * dist_mi + data / bw
		return time, dist_mi

		
	@staticmethod
	def estimate_query_comm_time(client_lat, client_lng, dc_id, data):
		dc = Comm.dc.loc[dc_id]
		dist_mi = LatLngUtil.compute_dist_mi(client_lat, client_lng,
											 dc.lat, dc.lng)
		time = Comm.sigma1 + Comm.sigma2 * dist_mi + data / Comm.bw_mobile
		return time, dist_mi


class Exec:
	@staticmethod
	def set_params(dc, beta1, beta2, gamma1, gamma2, theta1, theta2, latency):
		Exec.dc = dc
		Exec.beta1 = beta1
		Exec.beta2 = beta2
		Exec.gamma1 = gamma1
		Exec.gamma2 = gamma2
		Exec.theta1 = theta1
		Exec.theta2 = theta2
		Exec.latency = latency

		
	@staticmethod
	def estimate_map_time(dc_id, data):
		m = Exec.dc.loc[dc_id].m
		return data / (Exec.beta1 + Exec.beta2 * m)

	
	@staticmethod
	def estimate_reduce_time(dc_id, data):
		m = Exec.dc.loc[dc_id].m
		return data / (Exec.gamma1 + Exec.gamma2 * m)

	
	@staticmethod
	def estimate_query_time(dc_id, queries_per_sec):
		m = Exec.dc.loc[dc_id].m
		return Exec.latency * (queries_per_sec / (Exec.theta1 + Exec.theta2 * m))
