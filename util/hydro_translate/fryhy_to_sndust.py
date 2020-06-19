'''
Ezra S. Brooker

GET ALL OF THE OPEN RELEASE STUFF FOR HERE

LANL & NMC

DATE CREATED: 09 June 2020

LAST MODIFIED: 09 June 2020

This Python script takes the Chris Fryer ccSN hydrocode data and converts
it into an sndust Particle class object friendly format.

See "particle.py" with the main sndust open release code for details on it.

This script is intended for Python version 3

The data structure saved to the h5py file is as follows per Particle:

    sid: 
    times: time frames of hydro data
    temperatures: temparture frames
    densities: density frames
    position: position frames
    mass: mass of zone
    volume: volume of zone
    composition: Dict[str, float] = field(default_factory=dict)

'''

import os, sys, re, ast, json, argparse, struct
import numpy as np
import h5py as h5
import periodictable as pt
import matplotlib.pyplot as plt

from scipy.io import FortranFile
from scipy import optimize


from IPython import embed
# for handling Fortran binaries
hsize = struct.calcsize('@idddd')

# conversion constants for fryhy units to cgs units 
urad, utim, utmp, umas = 1.0E9, 1.0E1, 1.0E9, 1.989E33
ueng = urad**2 / utim**2
uprs = umas / (urad * utim**2)
uerg = umas * ueng
uden = umas / (urad**3)
uvel = urad / utim

fourpiover3 = (4.0*np.pi)/3.0



#====================================================================================#

def cca(inarr):
	''' cell center averaging for cell interface data structures '''
	return 0.5*(inarr[1:] + inarr[:-1])


def convert_record(data):
	''' for correctly loading Fortran binaries data (special record heades/footers)'''

	nc, time = struct.unpack('=id', data[4:16])

	start = hsize
	siz = 8 * nc
	form = '={0}d'.format(nc)
	sizp1  = 8 * (nc + 1)
	formp1 = '={0}d'.format(nc+1)

	x = urad * np.asarray(struct.unpack(formp1, data[start:start + sizp1]))
	start += sizp1
	v = uvel * np.asarray(struct.unpack(formp1, data[start:start + sizp1]))
	# skip q, dq
	start += 3 * siz
	u = ueng * np.asarray(struct.unpack(form, data[start:start + siz]))
	start += siz
	deltam = umas * np.asarray(struct.unpack(form, data[start:start + siz]))
	# skip abar
	start += 2 * siz
	rho = uden * np.asarray(struct.unpack(form, data[start:start + siz]))
	start += siz
	temp = utmp * np.asarray(struct.unpack(form, data[start:start + siz]))
	# skip tempe, ye
	start += 3 * siz
	pr = uprs * np.asarray(struct.unpack(form, data[start:start + siz]))

	xc = cca(x)
	vc = cca(v)
	volume = fourpiover3 * (xc**3.0)
	mass = volume * rho

	return dict({'time':time*utim, 'radii':xc, 'velocity':vc, 'eint':u, 'dmass':deltam, 
		'density':rho, 'temperature':temp, 'pressure':pr, 'mass':mass, 'volume':volume})


def collect_element_abundances(xiso, iso_names):
	#iso_dict = {}
	elem_dict = {}

	for idx, iso in enumerate(iso_names):
		elem = re.sub('[0-9]+','',iso)
		
		if elem == 'NEUT':
			elem = 'NEUTRONS'
		elif elem == 'PROT':
			elem = 'H'
			
		try:
			elem_dict[elem] += xiso[:,idx].copy()

		except:
			elem_dict[elem] = xiso[:,idx].copy()

	return elem_dict


def load_hydro_model_info(modelsf, gid):
	''' loads fryhy hydro model header data from "Rosetta Stone" file '''

	lines = []
	types = [np.str, np.str, np.str, np.float64, np.float64, np.float64, np.float64, \
			 np.float64, np.float64, np.float64, np.int]

	with open(modelsf, 'r') as f:
		for line in f:
			lines.append(line.expandtabs().strip().split(','))
	
	header = lines[0]
	lines  = np.array(lines[1:])
	lines  = np.char.replace(lines, 'NA', 'NaN')

	model_dict = {}
	for kdx, key in enumerate(header):
		model_dict[key] = lines[:,kdx].astype(types[kdx])

	return dict({'types': types, 'model_dict':model_dict})


def load_init_hydro_data(ccsnf, gid):
	''' fetches information and compositions from initial hydro starter master file '''

	# open master file for fryhy model inputs
	mfile = h5.File(ccsnf, 'r')
	mkeys = list(mfile.keys())
	dset  = mfile[mkeys[gid]]
	nc    = dset['radius'].size - 1

	aiso, ziso, xiso = dset['aiso'][()], dset['ziso'][()], dset['xiso'][()]
	isos = dset['iso_names'][()].astype(np.str)
	isos = np.char.replace(isos,' ','')
	mfile.close()
	xiso = cca(xiso)

	# collect the elemental abundances from the isotope abundances
	elems = collect_element_abundances(xiso, isos)

	elky = np.array(list(elems.keys()))
	cxiso = np.array(list(elems.values()))
	for el in ['H',"HE","C","O","MG","SI","NI"]:
		idx = np.where(elky==el)[0][0]
		plt.semilogy(cxiso[idx,:])
	plt.savefig('/Users/ebrooker/Desktop/test1.png', dpi=1200)

	return nc, dict({'aiso' : aiso, 'ziso' : ziso, 'xiso' : xiso, 'iso_names':isos}), elems


def load_outbin_hydro_data(binf, numcells):
	''' Load the output binary file from hydro model run '''

	# dump binary file contents into "fdata" var
	with open(binf, 'rb') as f:
		fdata = f.read()

	# get some information from top line of binary file
	recsize, bincells, time = struct.unpack('@iid', fdata[:16])
	blocksize = recsize + 8
	nrec      = int(len(fdata)/blocksize)
	frames    = np.linspace(0,nrec-1,nrec)
	lines     = frames.shape[0]



	# for extracting data per timestamp
	def get_frame(i):
		data = fdata[i * blocksize : (i+1) * blocksize]
		stack = convert_record(data)
		return stack

	hydat = get_frame(0)

	times = np.zeros(lines)
	temps = np.zeros((lines,numcells))
	rhos  = np.zeros_like(temps)
	rads  = np.zeros_like(temps)
	vols  = np.zeros_like(temps)
	mass  = np.reshape(hydat['mass'][:numcells].copy(), (1,numcells))

	for k in range(lines):
		hydat = get_frame(k)
		rads[k,:]  = hydat['radii'][:numcells].copy()
		vols[k,:]  = hydat['volume'][:numcells].copy()
		rhos[k,:]  = hydat['density'][:numcells].copy()
		temps[k,:] = hydat['temperature'][:numcells].copy()
		times[k]   = hydat['time']

	return times, dict({'temperature':temps, 'density':rhos, 
				'radius':rads, 'volume':vols, 'mass':mass})

 
def save_particle_data(savef, gid, mdl_info, elems, hydros, nc, time):
	''' takes all of the data and saves to HDF5 file as a set of particles '''

	model_dict = mdl_info['model_dict']
	types = mdl_info['types']

	tempname = savef.split('.')
	savef = '{0}{1}_.{2}'.format(tempname[0], model_dict['run'][gid], tempname[1])

	els = str(list(elems.keys()))
	abu = np.array(list(elems.values()))

	root_grp = h5.File(savef, 'w')
	info_grp = root_grp.create_group('model_info')

	kdx = 0
	mid = np.where(model_dict['num']==gid)[0][0]
	for key,val in model_dict.items():
		if types[kdx]=='S10' or types[kdx]==np.str:
			info_grp.create_dataset(key, data=np.string_(val[mid]))
		else:
			info_grp.create_dataset(key, data=val[mid], dtype=types[kdx])
		kdx+=1

	part_grp = root_grp.create_group('particles')

	for pid in range(nc):
		pid_grp = part_grp.create_group(str(pid))
		pid_grp.create_dataset('time', data=time)
		for key,val in hydros.items():
			pid_grp.create_dataset(key, data=val[:,pid])

		pid_grp.create_dataset('elements', data=els)
		pid_grp.create_dataset('abundances', data=abu[:,pid])

	root_grp.close()
	print('[I/O] successfully saved model {0} to HDF5 file'.format(model_dict['run'][gid]))
	print('      see new dust input deck at {0}'.format(savef))
	return savef

def create_sndust_input_deck(models_file, ccsn_file, hydro_file, save_file, gid):
	''' takes the fryhy out.bin data and converts to a format for hdf5 '''

	model_dict = load_hydro_model_info(models_file, gid)

	# get number of cells of original star data, composition data, etc
	nc, isos_dict, elem_dict = load_init_hydro_data(ccsn_file, gid)

	# load in the output data from Fortran binary (t,T,rho,r,V,M)
	time, hydro_dict = load_outbin_hydro_data(hydro_file, nc)

	new_savef = save_particle_data(save_file, gid, model_dict, elem_dict, hydro_dict, nc, time)
	
	print('DONE')

	return new_savef


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("models_file", type=str, help="model info textfile")
	parser.add_argument("ccsn_file", type=str, help="hdf5 input file")
	parser.add_argument("hydro_file", type=str, help="hydro run file")
	parser.add_argument("save_file", type=str, help="save file name for input deck")
	parser.add_argument("gid", type=int, help="mnodel ID number")

	args = parser.parse_args()

	new_save_file = create_sndust_input_deck(args.models_file, args.ccsn_file, args.hydro_file, args.save_file, args.gid)

	hf = h5.File(new_save_file,'r')
	pf = hf['particles']
	pk = list(pf.keys())

	elems = pf[pk[1]]['elements'][()]
	elems = np.array(ast.literal_eval(elems))

	xiso = np.zeros((85,1956))
	for i,p in enumerate(pk):
		abu = pf[p]['abundances'][()]
		xiso[:,i] = abu

	for el in ['H',"HE","C","O","MG","SI","NI"]:
		idx = np.where(elems==el)[0][0]
		plt.semilogy(xiso[idx,:])
	plt.savefig('/Users/ebrooker/Desktop/test2.png', dpi=1200)
