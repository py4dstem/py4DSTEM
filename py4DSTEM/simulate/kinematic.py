import numpy as np
import pymatgen as mg
from tqdm import tqdm

from ..process.rdf import single_atom_scatter
from ..process.utils import electron_wavelength_angstrom
from ..file.datastructure import PointList, PointListArray


class KinematicLibrary:
	"""
	Kinematic performs kinematic (single scattering) diffraction simulations 
	to construct a library of simulated PointLists with associated orientations, used for classifying.
	"""

	def __init__(self, structure, max_index=6, poles=None, voltage=300_000, tol_zone=0.1, tol_int=10,
		thickness=500, **kwargs):
		'''
		Create a Kinematic simulation object. 

		Accepts:
		structure		a pymatgen Structure object for the material
						or a string containing the Materials Project ID for the 
						structure (requires API key in config file, see:
						https://pymatgen.org/usage.html#setting-the-pmg-mapi-key-in-the-config-file

		max_index		maximum hkl indices to compute structure factors

		poles			numpy array (n,3) containing [h,k,l] indices for the n
						orientations required in the simulation.

		voltage			electron kinetic energy in Volts

		tol_zone		cutoff for discarding reciprocal lattice points that
						do not exactly satisfy Weiss zone law	

		tol_int			cutoff for excluding very weak structure factors

		thickness		sample thickness in Å
		'''

		if isinstance(structure,str):
			structure = mg.get_structure_from_mp(structure)

		assert isinstance(structure,mg.core.Structure), "structure must be pymatgen Structure object"

		sga = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)
		self.structure = sga.get_conventional_standard_structure()
		self.struc_dict = self.structure.as_dict()
		self.recip_lat = self.structure.lattice.reciprocal_lattice_crystallographic.abc

		print('Conventional Standard Structure used for calculation:')
		print(self.structure, flush=True)

		# hold on to a scattering factor calculator object
		self.scat_fac = single_atom_scatter()

		self.max_index = max_index
		self.tol_zone = tol_zone
		self.tol_int = tol_int

		self.a = self.structure.lattice.abc[0]
		self.N = thickness//self.a

		self.λ = electron_wavelength_angstrom(voltage)

		# rotation matrix to get a nonparallel vector for later calculations
		# rotates by a small amount about each axis
		c = np.cos(0.1/180*np.pi)
		s = np.sin(0.1/180*np.pi)
		self.R = np.array([[1,0,0],[0,c,-s],[0,s,c]]) @ \
			np.array([[c,0,s],[0,1,0],[-s,0,c]]) @ \
			np.array([[c,-s,0],[s,c,0],[0,0,1]])

		# ------- run computations --------
		self._compute_structure_factors()

		# if no poles are given, use the cubic structure's symmetric wedge:
		if poles is None:
			if 'n_poles' in kwargs.keys():
				n = kwargs['n_poles']
			else:
				n = 250 # ROUGHLY this many poles
			self.poles =self._cubic_poles(n)
		else:
			self.poles = poles

		self._run_simulation()


	def explore_library(self):
		from ipywidgets import interactive
		from IPython.display import display
		import matplotlib.pyplot as plt

		def f(x):
		    pl = self.pattern_library.get_pointlist(x,0)
		    plt.figure(2,figsize=(6,6))
		    plt.scatter(pl.data['qx'],pl.data['qy'],s=150*np.sqrt(pl.data['intensity']))
		    plt.axis('equal')
		    plt.title(f"g = [{pl.tags['pole'][0]:.0f},{pl.tags['pole'][1]:.0f},{pl.tags['pole'][2]:.0f}]")
		    for i in range(pl.length):
		        plt.text(pl.data['qx'][i]+0.008,pl.data['qy'][i],f"({pl.data['h'][i]:.0f},{pl.data['k'][i]:.0f},{pl.data['l'][i]:.0f})")
		        
		interactive_plot = interactive(f,x=(0,self.pattern_library.shape[0]-1))
		interactive_plot.children[-1].layout.height = '400px'
		display(interactive_plot)

# ---------- PRIVATE METHODS ---------------- #

	def _run_simulation(self):
		# allocate pointlistarray:
		tags = {'hkl':None, 'pole':None}
		coordinates = ('qx','qy','h','k','l','intensity')
		pla = PointListArray(coordinates=coordinates,
			shape=(self.poles.shape[0],1),tags=tags)

		# loop over all the poles
		for i in tqdm(range(self.poles.shape[0]),desc='Generating diffraction patterns'):
			pl = pla.get_pointlist(i,0)

			uvw = self.poles[i,:]

			# get the diffraction intensities for each reflection
			pattern = self._generate_pattern(uvw)

			# project reflections onto the plane defined by uvw
			proj_points = pattern[:,0:3] - ( (pattern[:,0:3] @ uvw) /np.sum(uvw**2))[:,np.newaxis]

			# generate two basis vectors in the plane:
			# first, arbitrarily rotate the pole to get a non-parallel vector, then take 
			# cross product with the pole
			x = np.cross(uvw, uvw @ self.R)
			# then cross this with the pole to get another ortho basis vector
			y = np.cross(uvw,x)
			# set to unit length
			x /= np.linalg.norm(x)
			y /= np.linalg.norm(y)

			# find the projections along these bases to get the diffraction spots
			qx = proj_points @ x
			qy = proj_points @ y

			data = np.vstack((qx,qy,pattern.T)).T

			pl.add_unstructured_dataarray(data)
			pl.tags['hkl'] = uvw
			pl.tags['pole'] = mg.core.lattice.get_integer_index(uvw,verbose=False)

		self.pattern_library = pla



	def _compute_structure_factors(self):
		# compute Fhkl for all reflections

		hh, kk, ll = np.mgrid[-self.max_index:self.max_index+1,\
			-self.max_index:self.max_index+1,-self.max_index:self.max_index+1]

		hklI = np.vstack((hh.ravel(),kk.ravel(),ll.ravel(),np.zeros((len(hh.ravel()),)))).T

		# make a separate datastructure to hold the complex structure factors
		hklF = np.zeros((len(hh.ravel()),),dtype=[('hkl','3int'),('F','complex')])

		# remove (0,0,0) **This is now handled down below
		#hklI = hklI[~np.all(hklI == 0, axis=1)]

		# loop over reciprocal lattice points
		for i in tqdm(range(hklI.shape[0]),desc='computing structure factors'):
			hkl = hklI[i,0:3].copy()

			q = 0.0 if np.all(hkl==0) else np.array([1/self.structure.lattice.d_hkl(hkl)])

			Fhkl = np.zeros((1,),dtype=np.complex)

			for site in self.struc_dict['sites']:
				Z = np.array([elements[site['species'][0]['element']]])
				self.scat_fac.get_scattering_factor(Z,np.array([1]),q,'A')
				F = self.scat_fac.fe

				Fhkl += F * np.exp( -2*np.pi*1j * (hkl @ site['abc']) )

			hklI[i,3] = np.abs(Fhkl)**2

			hklF['hkl'][i] = hkl
			hklF['F'][i] = Fhkl

		self.hklI = hklI
		self.hklF = hklF


	def _generate_pattern(self,uvw,a=None,N=None):
		# apply zone law to find reciprocal lattice points that may be excited
		# and scale structure factor by shape factor using excitation error
		pattern = np.zeros((1,4))

		hklI = self.hklI

		# find unit vector along the zone
		uvw_0 = uvw / np.sqrt(np.sum(uvw**2))

		k0 = uvw_0 / self.λ 
		
		for i in range(hklI.shape[0]):
			P = uvw_0 @ hklI[i,:3]
			if (np.abs(P) < self.tol_zone) and (hklI[i,3]>self.tol_int):
				#set_trace()
				refl = hklI[i,:].copy()

				# this is not quite the right excitation error, need line distance formula
				g = k0 + (refl[:3] * self.recip_lat)
				exc = (1/self.λ) - np.linalg.norm(g)

				refl[3] *= self._shape_factor(exc,self.a,self.N)
				if not np.isnan(refl[3]):
					pattern = np.vstack((pattern,refl))
				
		maxInt = np.nanmax(pattern[:,3])
		pattern[0,3] = maxInt
		pattern[:,3] /= maxInt

		return pattern

	def _generate_transmission(self,uvw,a,N):
		# this is the same as _generate_pattern but returns the complex
		# transmission function instead of the instensity
		pattern = np.zeros((0,),dtype=self.hklF.dtype)

		hklF = self.hklF

		# find unit vector along the zone
		uvw_0 = uvw / np.sqrt(np.sum(uvw**2))

		k0 = uvw_0 / self.λ 
		
		for i in range(hklF.shape[0]):
			P = uvw_0 @ hklF['hkl'][i]
			if (np.abs(P) < self.tol_zone) and (np.abs(hklF['F'][i])**2>self.tol_int):
				#set_trace()
				refl = hklF[i].copy()

				# this is not quite the right excitation error, need line distance formula
				g = k0 + (refl['hkl'] * self.recip_lat)
				exc = (1/self.λ) - np.linalg.norm(g)

				# IS THIS RIGHT?
				#refl['F'] = np.real(refl['F']) * self._shape_factor(exc,a,N) + 1j*np.imag(refl['F'])
				refl['F'] *= self._shape_factor(exc,a,N)
				if not np.isnan(refl['F']):
					pattern = np.hstack((pattern,refl))
			
		return pattern

	def _generate_dynamical_pattern(self,uvw):
		# account for multiple scattering by computing S-matrix for a unit cell
		# thick slab and applying it iteratively, along with a propagation operator
		pass

	def _shape_factor(self,s,a,N):
		"""
		find SS*(s) for excitation error s, unit cell size a, number atoms N
		"""
		if np.isclose(s,0,atol=1e-10):
			return N**2
		else:
			return np.sin(np.pi*s*a*N)**2 / np.sin(np.pi*s*a)**2

	def _cubic_poles(self,n):
		print('Using cubic symmetric poles...', flush=True)
		# generate n diffraction vectors spaced (uniformly?) within the symmetrically unique 
		# subset of cubic diffraction vectors
		a = np.array([1,0,0])
		b = np.array([1,1,0])
		c = np.array([1,1,1])

		a0 = a / np.linalg.norm(a)
		b0 = b / np.linalg.norm(b)
		c0 = c / np.linalg.norm(c)

		# matrix of unit vectors along symmetric poles
		abc = np.vstack((a0,b0,c0))

		n = np.ceil(np.cbrt(n))

		i,j,k = np.mgrid[0:n,0:n,0:n]
		ijk = np.vstack((i.ravel(),j.ravel(),k.ravel())).T
		ijk = ijk[~np.all(ijk == 0, axis=1)]
		ijk0 = ijk / np.linalg.norm(ijk,axis=1)[:,np.newaxis]
		ijk0 = np.unique(ijk0,axis=0)

		return ijk0 @ abc



# a dictionary for converting element names into Z
els = ('H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si',\
	   'P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni',\
	   'Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb',\
	   'Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',\
	   'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho',\
	   'Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl',\
	   'Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am',\
	   'Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt',\
	   'Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og','Uue','Ubn','Ubn')

elements = {els[i]: i+1 for i in range(len(els)) }
