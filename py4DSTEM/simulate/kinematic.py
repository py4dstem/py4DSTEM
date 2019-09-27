import numpy as np
import pymatgen as mg
from tqdm import tqdm

from ..process.rdf import single_atom_scatter
from ..process.utils import electron_wavelength_angstrom

from pdb import set_trace

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

class Kinematic:
	"""
	Kinematic performs kinematic diffraction simulations, to construct a library of simulated PointLists
	with associated orientations, used for classifying.
	"""

	def __init__(self, structure, max_index=6, poles=None, voltage=300_000, tol_zone=0.1, tol_int=10,
		thickness=500):
		'''
		Create a Kinematic simulation object. 
		Accepts:
		structure		a pymatgen Structure object for the material
						can also pass string of materials project ID (requires API key)
		max_index		maximum hkl indices to compute
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
		self.N = thicness//self.a

		self.λ = electron_wavelength_angstrom(voltage)

		# ------- run computations --------
		self.compute_structure_factors()


	def compute_structure_factors(self):
		# generate diffraction patterns and convert to PointListArray

		hh, kk, ll = np.mgrid[-self.max_index:self.max_index,\
			-self.max_index:self.max_index,-self.max_index:self.max_index]

		hklI = np.vstack((hh.ravel(),kk.ravel(),ll.ravel(),np.zeros((len(hh.ravel()),)))).T

		# remove (0,0,0)
		hklI = hklI[~np.all(hklI == 0, axis=1)]

		# loop over reciprocal lattice points
		for i in tqdm(range(hklI.shape[0]),desc='computing structure factors'):
			hkl = hklI[i,0:3].copy()

			q = np.array([1/self.structure.lattice.d_hkl(hkl)])

			Fhkl = np.zeros((1,),dtype=np.complex)

			for site in self.struc_dict['sites']:
				Z = np.array([elements[site['species'][0]['element']]])
				self.scat_fac.get_scattering_factor(Z,np.array([1]),q,'A')
				F = self.scat_fac.fe

				Fhkl += F * np.exp( -2*np.pi*1j * (hkl @ site['abc']) )

			hklI[i,3] = np.abs(Fhkl)**2

		self.hklI = hklI

		#return self.hklI

	def _generate_pattern(self,uvw):
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

	def _shape_factor(self,s,a,N):
	    """
	    find SS*(s) for excitation error s, unit cell size a, number atoms N
	    """
	    return np.sin(np.pi*s*a*N)**2 / np.sin(np.pi*s*a)**2



