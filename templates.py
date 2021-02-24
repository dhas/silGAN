import numpy as np

def takeoff(num=10, k=0.1, len=512):
	sg = np.full((num, 2, len), np.nan)	
	for i, l in enumerate(np.linspace(0, 0.9, num=num)):
		n_steps = np.int(np.ceil(l/k))
		for n in range(1):
			sg[i, n] = np.zeros(len)			
			mid = len//2
			sg[i, n, mid:] = l			
			sg[i, n, mid-n_steps:mid] = np.linspace(0, l, num=n_steps)
			# sg[i, n, mid-50:256] = np.linspace(0, l, num=50)
	return sg

def stop_takeoff(num=10, k=0.1, len=512):
	sg = np.full((num, 2, len), np.nan)
	for i, l in enumerate(np.linspace(0, 0.9, num=num)):
		slope = np.int(np.ceil(l/k))
		parts = np.array([0, 64, 256, 512]) #np.linspace(0, len, num=4).astype(int)		
		for n in range(1):
			sg[i, n] = np.zeros(len)
			#come to a stop
			sg[i, n, parts[0]: parts[1]-slope] = l
			sg[i, n, parts[1]-slope: parts[1]] = np.linspace(l, 0, num=slope)

			#take off
			sg[i, n, parts[2]: parts[2]+slope] = np.linspace(0, l, num=slope)
			sg[i, n, parts[2]+slope: parts[3]] = l			
	return sg
