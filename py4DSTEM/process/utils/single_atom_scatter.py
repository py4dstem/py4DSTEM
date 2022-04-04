import numpy as np
import os

class single_atom_scatter(object):
    """
    This class calculates the composition averaged single atom scattering factor for a
    material. The parameterization is based upon Lobato, Acta Cryst. (2014). A70,
    636–649.

    Elements is an 1D array of atomic numbers.
    Composition is a 1D array, same length as elements, describing the average atomic
    composition of the sample. If the Q_coords is a 1D array of Fourier coordinates,
    given in inverse Angstroms. Units is a string of 'VA' or 'A', which returns the
    scattering factor in volt angtroms or in angstroms.
    """
    def __init__(self,elements=None,composition=None,q_coords=None,units=None):
        self.elements = elements
        self.composition = composition
        self.q_coords = q_coords
        self.units = units
        path = os.path.join(os.path.dirname(__file__),'scattering_factors.txt')
        self.e_scattering_factors = np.loadtxt(path,dtype=np.float)

        return

    def electron_scattering_factor(self,Z,gsq,units='A'):
        ai = self.e_scattering_factors[Z-1,0:10:2]
        bi = self.e_scattering_factors[Z-1,1:10:2]
        # gsq = g**2
        #Planck's constant in Js
        h= 6.62607004e-34
        #Electron rest mass in kg
        me = 9.10938356e-31
        #Electron charge in Coulomb
        qe = 1.60217662e-19
        #Electron wave number (reciprocal of wavelength) in Angstrom
        # k0 = wavev(E)
        #Relativistic electron mass correction
        # gamma = relativistic_mass_correction(E)
        # return 2*np.pi*gamma*me*qe/k0/h/h*1e-20

        fe = np.zeros_like(gsq)
        a0=0.529177
        eps0= 8.85418782e-42
        qe= 1.60217e-19
        for i in range(5):
            fe +=ai[i]*(2+bi[i]*gsq)/(1+bi[i]*gsq)**2

        # Result can be returned in units of Volt Angstrom ('VA') or Angstrom ('A')
        if units=='VA': return h**2/(2*np.pi*me*qe)*1e18*fe
        elif units == 'A' : return fe

    def get_scattering_factor(self,elements=None,composition=None,q_coords=None,units=None):
        if elements is None:
            assert(not self.elements is None), "Must pass a list of atomic numbers in either class initialization or in call to get_scattering_factor()"
            elements = self.elements

        if composition is None:
            assert(not self.elements is None), "Must pass composition fractions in either class initialization or in call to get_scattering_factor()"
            composition = self.composition

        if q_coords is None:
            assert(not self.elements is None), "Must pass a q_space array in either class initialization or in call to get_scattering_factor()"
            q_coords = self.q_coords

        if units is None:
            units = self.units
            if(self.units is None):
                print("Setting output units to Angstroms")
                units = 'A'

        assert(len(elements)==len(composition)), "Each element must have an associated composition."

        if np.sum(composition) > 1:
            #normalize composition if passed as stoichiometry instead of atomic fractions
            composition /= np.sum(composition)

        fe = np.zeros_like(q_coords)
        for i in range(len(elements)):
            fe += composition[i]*self.electron_scattering_factor(elements[i],np.square(q_coords),units)

        self.fe = fe

