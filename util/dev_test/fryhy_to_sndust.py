'''
Ezra S. Brooker

LANL & NMC

DATE CREATED: 08 June 2020

LAST MODIFIED: 12 June 2020

This Python script takes the Chris Fryer ccSN hydrocode data and converts
it into an sndust Particle class object friendly format.

See "particle.py" with the main sndust open release code for details on it.

This script is intended for Python version 3

The data structure saved to the h5py file is as follows:

	'model_info' <-- subgroup
		ALL THE FIELDS FROM THE ROSETTA STONE FILE
		run
		Mprog
		Einj
		Eexp
		etc etc

	'particles'  <-- subgroup
		DATA SAVED PER PARTICLE
		    times: time frames of hydro data
		    temperatures: temparture frames
		    densities: densities frames
		    position: position frames
	    	mass: mass of zone
	    	volume: volume of zone
	    	
	    	composition <-- subgroup
	    		each element is saved as a single number in a
				create_dataset instance with the element name as
				the dataset keyname

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

	return dict({'time':time, 'position':xc, 'velocity':vc, 'eint':u, 'dmass':deltam, 
		'densities':rho, 'temperatures':temp, 'pressure':pr, 'mass':mass, 'volume':volume})


def collect_element_abundances(xiso, iso_names):
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

	print('[LOADING] reading hydro model Rosetta Stone info')
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
	print('[LOADING] reading initial hydro model')
	# open master file for fryhy model inputs
	mfile = h5.File(ccsnf, 'r')
	mkeys = list(mfile.keys())
	dset  = mfile[mkeys[gid]]
	nc    = dset['radius'].size - 1

	xiso = dset['xiso'][()]
	isos = dset['iso_names'][()].astype(np.str)
	isos = np.char.replace(isos,' ','')
	mfile.close()
	xiso = cca(xiso)

	# collect the elemental abundances from the isotope abundances
	print('[LOADING] collecting isotopic abundances into elements')
	elems = collect_element_abundances(xiso, isos)
	return nc, elems


def load_outbin_hydro_data(binf, numcells):
	''' Load the output binary file from hydro model run '''
	print('[LOADING] reading hydro time evolution data')
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

	# fill in rows for each timestamp frame
	for k in range(lines):
		hydat = get_frame(k)
		rads[k,:]  = hydat['position'][:numcells].copy()
		vols[k,:]  = hydat['volume'][:numcells].copy()
		rhos[k,:]  = hydat['densities'][:numcells].copy()
		temps[k,:] = hydat['temperatures'][:numcells].copy()
		times[k]   = hydat['time']

	return times, dict({'temperatures':temps, 'densities':rhos, 
				'radius':rads, 'volume':vols, 'mass':mass})

 
def save_particle_data(savef, gid, mdl_info, elems, hydros, nc, time):
	''' takes all of the data and saves to HDF5 file as a set of particles '''

	model_dict = mdl_info['model_dict']
	types = mdl_info['types']

	tempname = savef.split('.')
	savef = '{0}{1}_.{2}'.format(tempname[0], model_dict['run'][gid], tempname[1])

	print('[WRITING] opening HDF5 file: {0}'.format(savef))

	root_grp = h5.File(savef, 'w')	

	# save Fryer ccSN model info in a separate subgroup
	kdx = 0
	mid = np.where(model_dict['num']==gid)[0][0]
	info_grp = root_grp.create_group('model_info')
	print('[WRITING] saving hydro model info to HDF5 in "model_info" subgroup')
	for md_key,md_val in model_dict.items():
		if types[kdx]=='S10' or types[kdx]==np.str:
			info_grp.create_dataset(md_key, data=np.string_(md_val[mid]))
		else:
			info_grp.create_dataset(md_key, data=md_val[mid], dtype=types[kdx])
		kdx+=1


	# create a subgroup for the particles, save the particle data for each particle
	part_grp = root_grp.create_group('particles')
	print('[WRITING] saving particle data to HDF5 in "particles" subgroup')
	for pid in range(nc):
		pid_grp = part_grp.create_group(str(pid))
		pid_grp.create_dataset('times', data=time)
		
		# hydro data
		for hy_key,hy_val in hydros.items():
			pid_grp.create_dataset(hy_key, data=hy_val[:,pid])

		# element composition, most sensible way to save, I think
		# HDF5 and arrays of keys and dictionaries don't play nicely
		abu_grp = pid_grp.create_group('composition')
		for el_key,el_val in elems.items():
			abu_grp.create_dataset(el_key, data=el_val[pid])

	root_grp.close()

	print('[WRITING] successfully saved model {0} to HDF5 file'.format(model_dict['run'][gid]))
	print('          see new dust input deck --> {0}'.format(savef))
	
	return savef


def create_sndust_input_deck(models_file, ccsn_file, hydro_file, save_file, gid):
	''' takes the fryhy out.bin data and converts to a format for hdf5 '''

	model_dict = load_hydro_model_info(models_file, gid)

	# get number of cells of original star data, composition data, etc
	nc, elem_dict = load_init_hydro_data(ccsn_file, gid)

	# load in the output data from Fortran binary (t,T,rho,r,V,M)
	time, hydro_dict = load_outbin_hydro_data(hydro_file, nc)

	new_savef = save_particle_data(save_file, gid, model_dict, elem_dict, hydro_dict, nc, time)

	print('[FINISHED]')
	return new_savef


#====================================================================================#


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("models_file", type=str, help="model info textfile")
	parser.add_argument("ccsn_file", type=str, help="hdf5 input file")
	parser.add_argument("hydro_file", type=str, help="hydro run file")
	parser.add_argument("save_file", type=str, help="save file name for input deck")
	parser.add_argument("gid", type=int, help="mnodel ID number")

	args = parser.parse_args()

	new_save_file = create_sndust_input_deck(args.models_file, args.ccsn_file, args.hydro_file, args.save_file, args.gid)




