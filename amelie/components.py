#!/usr/bin/env python

from __future__ import (print_function, division, absolute_import)

import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G, M_sun
import astropy.units as u
from collections import OrderedDict

from parameters import Parameter, _parameter_names#, LimbDarkening
from conversions import *

# companion.P.value, error etc needs to be used within the code, so the
# attributes error, etc should not be lost
# later when mcmc fit is done should set companion.P.dist to the distribution,
# so that one can call companion.P.std, companion.P.value
class Companion(object):
    def __init__(self, name=None):

        self.name  = name
        self.fname = ':'.join(['companion', name])

        # assign parameter names
        self.allowed_set_parameters = _parameter_names['allowed_set_parameters']
        self.allowed_lc_priors      = _parameter_names['allowed_lc_priors']
        self.allowed_rv_priors      = _parameter_names['allowed_rv_priors']

        self.parameter_names = tuple(set(self.allowed_lc_priors +
                                         self.allowed_rv_priors))
#        self.parameter_names = self.varied_parameters + self.derived_parameters

        for name in self.parameter_names:
#            print('creating ', name)
            setattr(self, '_'.join(['', name]), Parameter(name=name))

        # add default numerical values for given parameters
#        has_default_values = ('P', 'Tpri', 'cosi', 'J', 'f_s', 'f_c', 'q')

        self.set_parameter('P',        value=1)
        self.set_parameter('Tpri',     value=0)
        self.set_parameter('cosi',     value=0)
        self.set_parameter('J',        value=0)
        self.set_parameter('f_s',      value=0)
        self.set_parameter('f_c',      value=0)
        self.set_parameter('q',        value=1)
        self.set_parameter('light_3',  value=0)
        self.set_parameter('vsini_1',  value=0)
        self.set_parameter('lambda_1', value=0)
        self.set_parameter('dPdt',     value=0)
#        self.set_parameter('M_1',      value=1)


#        super(OrbitalParameters, self).__init__()


    def set_parameter(self, name, **kwargs):
        for key, val in kwargs.items():
            if key == 'value':
                key = '_'.join(['', key])
            setattr(getattr(self, '_'.join(['', name])), key, val)

    def _warp_times(self, t):
        """ warp times to account for changes in orbital period """
        return t - 0.5 * self.dPdt_dd * (t - self.Tpri.value)**2 / self.P.value

#    def _in_transit_and_same_night_mask(self, t, t_exp=None):
#        m = self._in_transit_mask(t, t_exp=t_exp)
#        day = int(np.mean(t[m]))
#        m |= np.array([np.asarray(d.x, dtype=int) == day], dtype=bool)
#
#        return 

    def _in_transit_mask(self, t, f=1.0, t_exp=None):
        p = (t - self.Tpri.value) % self.P.value / self.P.value
#        p[p> 0.5] -= 1
        pw = self.W.value / self.P.value

        p_exp = t_exp / self.P.value if t_exp is not None else 0

        # primary eclipse
#        pri_ecl = np.logical_and(p > -f*0.5*pw, p < f*0.5*pw)
        pri_ecl = np.logical_or(p > (1 - f*0.5*(pw - p_exp)), 
                                p < f*0.5*(pw + p_exp))

        # secondary eclipse
        pwsec = self.Wsec.value / self.P.value
        psec = (self.Tsec.value - self.Tpri.value) % self.P.value / self.P.value
        sec_ecl = np.logical_and(p > (psec - f*0.5*(pwsec - p_exp)),
                                 p < (psec + f*0.5*(pwsec + p_exp)))

        m = pri_ecl | sec_ecl
        
        return m

    def _get_n_int(self, t, oversample=1, in_transit=True, f=1.0, t_exp=None):
        """
        get n_int array for ellc.lc() input.
        """
#        print('time', t)
#        f /= 100 + 1
        m = np.zeros_like(t, dtype=int) + int(oversample)
        if in_transit:
#            print('tpri, p', self.Tpri.value, self.P.value)
#            p = (t - self.Tpri.value) % self.P.value / self.P.value
#            p[p> 0.5] -= 1
#            pw = self.W.value / self.P.value
##            print('rsum', self.rsum.value)
##            print('k', self.k.value)
##            print('roa', self.r_1.value)
##            print('b', self.b.value)
##            print('cosi', self.cosi.value)
##            print('incl', self.incl.value)
##            print('e', self.e.value)
##            print('width', self.W.value)
#            pri_ecl = np.logical_and(p > -f*0.5*pw, p < f*0.5*pw)
#
#            # secondary eclipse
#            psec = (self.Tsec.value - self.Tpri.value) / self.P.value
#            sec_ecl = np.logical_and(p > (psec - f*0.5*pw),
#                                     p < (psec + f*0.5*pw))
#
#            out = ~(pri_ecl | sec_ecl)
            out = ~self._in_transit_mask(t, f=f, t_exp=t_exp)
            m[out] = 0

#        print('total in transit', np.sum(m))
        if np.sum(m) == 0:
            return np.ones_like(t, dtype=int)
        return m


#    def _has_priors(self):
#        return tuple([p for p in self.parameter_names
#                        if getattr(getattr(self, p), 'bounds') is not None
#                        and if getattr(getattr(self, p), ''
                      
#        setattr(self, getattr(super(Companion, self).parameter, '_value'),
#                Parameter(parameter, value=value, error=error,
#                          bounds=bounds, gprior=gprior))
#        setattr(self, parameter,
#                Parameter(parameter, value=value, error=error,
#                          bounds=bounds, gprior=gprior))

#    def get_parameter_names(self):
#        return (p
#                for p in self.__dict__.keys()
#                if p.startswith('_'))


    # impact parameter
#    @b.setter
#    def b(self, val):
#        v = get_impact_parameter(self.cosi, self.r_1, self.e, self.w)
#        self.set_parameter('b', value=v)
    @property
    def light_3(self):
        return self._light_3

    @property
    def b(self):
#        print('omega', self.w.value)
        v = get_impact_parameter(self.cosi.value, self.r_1.value,
                                 self.e.value, np.deg2rad(self.w.value))
        self.set_parameter('b', value=v)
        return self._b

    @property
    def bsec(self):
#        print('omega', self.w.value)
        v = get_eclipse_impact_parameter(self.cosi.value, self.r_1.value,
                                 self.e.value, np.deg2rad(self.w.value))
        self.set_parameter('bsec', value=v)
        return self._bsec

    # orbital period
    @property
    def P(self):
        return self._P

    # change in orbital period
    @property
    def dPdt(self):
#        if self.log_dPdt.value is None:
#            v = 0
#        else:
#            v = np.exp(self.log_dPdt.value)
#        self.set_parameter('dPdt', value=v)
#        return self._dPdt
#        self.set_parameter('dPdt', value=v)
        return self._dPdt

    @property
    def dPdt_dd(self):
        return (self.dPdt.value * u.s / u.yr).to(u.d / u.d).value

##    @property
#    def log_dPdt(self):
#        return self._log_dPdt

    # radius ratio, r_2/r_1
    @property
    def k(self):
        return self._k

    # surface brightness ratio, J_2/J_1
    @property
    def J(self):
        return self._J
    
    # transit depth deterministic
    # self.D assumed to be Deterministic()
    # calling self.D will return self.D.value
#    @property
#    def D(self):
#        f = lambda k, J: k**2 * (1 - J)
#        return self.D.value(self.k.value, self.J.value)

    # transit depth
    @property
    def D(self):
        v = get_transit_depth(self.k.value, self.J.value)
        self.set_parameter('D', value=v)
        return self._D

    # orbital inclination
    @property
    def incl(self):
        v = get_inclination(self.cosi.value)
        self.set_parameter('incl', value=v)
        return self._incl

    # transit duration (width)
    @property
    def Wsec(self):
        v = get_eclipse_width(self.r_1.value, self.P.value, self.k.value,
                              self.bsec.value, self.e.value, self.w.value,
                              self.incl.value)
#        print('width', v)
        self.set_parameter('Wsec', value=v)
        return self._Wsec

    # transit duration (width)
    @property
    def W(self):
        v = get_transit_width(self.r_1.value, self.P.value, self.k.value,
                              self.b.value, self.e.value, self.w.value,
                              self.incl.value)
#        print('width', v)
        self.set_parameter('W', value=v)
        return self._W

    # cosine of orbital inclination
    @property
    def cosi(self):
        return self._cosi


    # scaled primary radius, R_1/a
    @property
    def r_1(self):
        v = get_radius_scaled_primary(self.rsum.value, self.k.value)
        self.set_parameter('r_1', value=v)
        return self._r_1

    # scaled secondary radius
    @property
    def r_2(self):
        v = get_radius_scaled_secondary(self.r_1.value, self.k.value)
        self.set_parameter('r_2', value=v)
        return self._r_2

    # scaled sum of radii
    @property
    def rsum(self):
        return self._rsum

    # time of primary eclipse
    @property
    def Tpri(self):
#        if self._Tsec.value is not None:
#            v = get_time_of_primary_eclipse(self.Tsec.value, self.e.value,
#                                            self.w.value, self.P.value,
#                                            self.incl.value)
#            self.set_parameter('Tpri', value=v)
        return self._Tpri

#        if self._Tpri.value is None and self._Tsec.value is not None:
#            v = get_time_of_primary_eclipse(self.Tsec.value, self.e.value,
#                                            self.w.value, self.P.value,
#                                            self.incl.value)
#            self.set_parameter('Tpri', value=v)
#        return self._Tpri

    # time of secondary eclipse
    @property
    def Tsec(self):
#        if self._Tsec.value is None and self._Tpri.value is not None:
        v = get_time_of_secondary_eclipse(self.Tpri.value, self.e.value,
                                          self.w.value, self.P.value,
                                          self.incl.value)
        self.set_parameter('Tsec', value=v)
        return self._Tsec

    @property
    def P(self):
        return self._P

    # semi-major axis of binary
    @property
    def a(self):
        v = _check_valid(get_semimajor_axis, self.a_1.value, self.q.value)
        self.set_parameter('a', value=v)
        return self._a

    # barycentric semi-major axis of primary
    @property
    def a_1(self):
        v = _check_valid(get_semimajor_axis_primary,
                         self.e.value, self.incl.value, self.Kpri.value,
                         self.P.value)
        self.set_parameter('a_1', value=v)
        return self._a_1

    @property
    def a_2(self):
        v = _check_valid(get_semimajor_axis_secondary, self.a.value, self.a_1.value)
        self.set_parameter('a_2', value=v)
        return self._a_2

    # eccentricity
    @property
    def e(self):
        v = get_eccentricity(self.f_s.value, self.f_c.value)
        self.set_parameter('e', value=v)
        return self._e

    # mass function
    @property
    def fM(self):
        v = get_mass_function(self.e.value, self.Kpri.value, self.P.value)
        self.set_parameter('fM', value=v)
        return self._fM

    # radial component of eccentricity, sqrt(e).sin(w)
    @property
    def f_s(self):
        return self._f_s

    # tangential component of eccentricity, sqrt(e).cos(w)
    @property
    def f_c(self):
        return self._f_c

    # radial velocity semi-amplitude
    @property
    def Kpri(self):
        return self._Kpri

    # log surface gravity of primary
    @property
    def log_g_1(self):
        v = get_surface_gravity_primary(self.fM.value, self.P.value,
                                        self.r_1.value, self.incl.value,
                                        self.q.value)
        self.set_parameter('log_g_1', value=v)
        return self._log_g_1

    # log surface gravity of secondary
    @property
    def log_g_2(self):
        v = get_surface_gravity_secondary(self.fM.value, self.P.value,
                                          self.r_2.value, self.incl.value)
        self.set_parameter('log_g_2', value=v)
        return self._log_g_2

    # mass of primary (solar units)
    @property
    def M_1(self):
        return self._M_1

    # mass of secondary (solar units)
    @property
    def M_2(self):
        if self.M_1.value is None:
            raise ValueError("""M_1 needs to be set to a value before M_2 can be
                             computed, e.g. `self._M_1 = 0.8`.
                             Warning: the mass calculation assumes M_2 << M_1,
                             which may be a bad approximation since you are
                             modelling a (single line) binary""")
        # assuming M_2 << M_1
        v = get_mass_secondary_solar(self.M_1.value, self.Kpri.value,
                                     self.e.value, self.incl.value,
                                     self.P.value)
        self.set_parameter('M_2', value=v)
        return self._M_2

    # mass of primary (jovian units)
    @property
    def Mjup_1(self):
        v = get_mass_primary_jupiter(self.M_1.value)
        self.set_parameter('Mjup_1', value=v)
        return self._Mjup_1

    # mass of secondary (jovian units)
    @property
    def Mjup_2(self):
        v = get_mass_secondary_jupiter(self.M_2.value)
        self.set_parameter('Mjup_2', value=v)
        return self._Mjup_2

    # mass of secondary (earth units)
    @property
    def Mearth_2(self):
        v = get_mass_secondary_earth(self.M_2.value)
        self.set_parameter('Mearth_2', value=v)
        return self._Mearth_2

    # mass ratio, M_2/M_1
    @property
    def q(self):
        if self.M_1.value is None:
            v = self._q.value
        else:
            v = get_mass_ratio(self.M_2.value, self.M_1.value)
        self.set_parameter('q', value=v)
        return self._q

    # density of primary
    @property
    def rho_1(self):
        """ uses M_2 << M_1 approximation, may be bad depending on the case """
        v = get_density_primary(self.M_1.value, self.q.value, self.P.value,
                                self.r_1.value)
        self.set_parameter('rho_1', value=v)
        return self._rho_1

    @property
    def rho_2(self):
        if self.M_1.value is None:
            raise ValueError("""M_1 needs to be set to a value before rho_2 can be
                             computed, e.g. `self._M_1 = 0.8`.
                             Warning: the calculation assumes M_2 << M_1,
                             which may be a bad approximation""")
        v = get_density_secondary(self.M_2.value, self.R_2.value)
        self.set_parameter('rho_2', value=v)
        return self._rho_2

    # primary radius (solar units)
    @property
    def R_1(self):
        v = _check_valid(get_radius_primary_solar, self.r_1.value, self.a.value)
#        v = get_radius_primary_solar(self.r_1.value, self.a.value)
        self.set_parameter('R_1', value=v)
        return self._R_1

    # secondary radius (solar units)
    @property
    def R_2(self):
        v = _check_valid(get_radius_secondary_solar, self.r_2.value, self.a.value)
#        v = get_radius_secondary_solar(self.r_2.value, self.a.value)
        self.set_parameter('R_2', value=v)
        return self._R_2

    # primary radius (jovian units)
    @property
    def Rjup_1(self):
        v = get_radius_primary_jupiter(self.R_1.value)
        self.set_parameter('Rjup_1', value=v)
        return self._Rjup_1

    # secondary radius (jovian units)
    @property
    def Rjup_2(self):
        v = get_radius_secondary_jupiter(self.R_2.value)
        self.set_parameter('Rjup_2', value=v)
        return self._Rjup_2

    # secondary radius (earth units)
    @property
    def Rearth_2(self):
        v = get_radius_secondary_earth(self.R_2.value)
        self.set_parameter('Rearth_2', value=v)
        return self._Rearth_2

    # argument of periastron
    @property
    def w(self):
        v = get_argument_of_periastron(self.f_s.value, self.f_c.value)
#        v = np.rad2deg(np.arctan2(self.f_s.value, self.f_c.value))
        self.set_parameter('w', value=v)
        return self._w

    # v.sini of primary star
    @property
    def vsini_1(self):
        return self._vsini_1

    # spin-orbit angle of primary star
    @property
    def lambda_1(self):
        return self._lambda_1


class Primary(object):
    def __init__(self, name=None, ld={}, ldc={}, shape='sphere'):
#        super(Primary, self).__init__()

        self._allowed_priors = ('ldc_1', 'ldc_2')
#        varied_parameters = ('ldc_1', 'ldc_2')

        self.channels = OrderedDict()

        # added 2020-07-19: bug where order of self.channels and self.ld were
        # different when adding multiple channels
#        self.ld       = OrderedDict()

#        self.gdc    = None
#        self.ld     = None
#        self._ldc   = None
        self.grid   = 'very_sparse'
        self.shape  = shape

#        for name in varied_parameters:
#            setattr(self, '_'.join(['', name]), Parameter(name=name))


#    def set_parameter(self, name, **kwargs):
#        for key, val in kwargs.items():
#            if key == 'value':
#                key = '_'.join(['', key])
#            setattr(getattr(self, '_'.join(['', name])), key, val)


    def add_channel(self, channel, ld):
        self.channels[channel] = ld#LimbDarkening(channel, ld)
        ld.fname = ':'.join(['channel', channel])

    def get_channel(self, channel=None):
        if channel is None:
            return self.channels
        return self.channels[channel]

    @property
    def ldc(self):
        return self._ldc

#class Primary(object):
#    def __init__(self):
#        super(Star, self).__init__()
#
#        self.gdc    = None
#        self.ld     = None
#        self._ldc   = None
#        self.grid   = 'very_sparse'
#
#    @property
#    def ldc(self):
#        return self._ldc
#
#
#
#class Primary(Star):
#    def __init__(self):
#
#        super(Primary, self).__init__()
#        
#        self._light_3 = 0.0
##        self._mass    = 1.0
#
#        # nuisance for primary in case of DL
#        self._jitter  = None
#        self._gamma   = None
#
#    # limb darkening parameters
#    @property
#    def ldc(self):
#        return super(Primary, self).ldc
#
#    # third light (dilution parameter)
#    @property
#    def light_3(self):
#        return self._light_3
#
#    # radial velocity jitter on primary
#    @property
#    def jitter(self):
#        return self._jitter
#
#    # radial velocity offset of primary (systemic velocity)
#    @property
#    def gamma(self):
#        return self._gamma
#
#
#class SecondarySL(Star, OrbitalParamsBinaryConvention, NuisanceParams):
#
#    def __init__(self):
#        super(SecondarySL, self).__init__()
#        
#class SecondaryDL(SecondarySL):
#    def __init__(self):
#        super(SecondaryDL, self).__init__()
#
#        self._K_2 = None
#        self._vsini_2 = None
#        self._lambda_2 = None
#
#    @property
#    def K_2(self):
#        return self._K_2
#
#    @property
#    def lambda_2(self):
#        return self._lambda_2
#
#    # mass of primary (solar units)
#    @property
#    def m_1(self):
#        return (1.0361e-7 * (1 - self.e**2)**1.5 * 
#                ((self.K_1 + self.K_2)/1e3)**2 * self.K_2/1e3 * self.P /
#                np.sin(np.deg2rad(self.incl))**3)
#
#    # mass of secondary (solar units)
#    @property
#    def m_2(self):
#        return (1.0361e-7 * (1 - self.e**2)**1.5 * 
#                ((self.K_1 + self.K_2)/1e3)**2 * self.K_1/1e3 * self.P / 
#                np.sin(np.deg2rad(self.incl))**3)
#    
#    @property
#    def q(self):
#        return self.K_1 / self.K_2
#
#    @property
#    def vsini_2(self):
#        return self._vsini_2
#
#
#class Planet(OrbitalParamsPlanetConvention, NuisanceParams):
#    def __init__(self):
#        super(Planet, self).__init__()
#
#
#
#
#if __name__ == '__main__':
##    sec = SecondarySL()
#    sec = Planet()
#    sec._P = 2.208
#    sec._f_s = 0.0
#    sec._f_c = 0.0
#    sec._b = 0.11
##    sec._cosi = np.cos(np.deg2rad(89.1))
##    sec._k = np.sqrt(0.00699)
#    sec._D = 0.00699
##    sec._rsum = 0.16
#    sec._W = 0.1129
#    sec._K_1 = 5200.0
##    sec._m_1 = 1.155
##    sec._K_2 = 17500.
#    print('period', sec.P)
#    print('tpri', sec.T_pri)
#    print('fs', sec.f_s)
#    print('fc', sec.f_c)
#    print('e', sec.e)
#    print('w', sec.w)
##    print('cosi', sec.cosi)
#    print('b', sec.b)
#    print('incl', sec.incl)
##    print('k', sec.k)
#    print('D', sec.D)
##    print('rsum', sec.rsum)
#    print('W', sec.W)
#    print('r1', sec.r_1)
#    print('R1', sec.R_1)
#    print('r2', sec.r_2)
#    print('R2', sec.R_2)
#    print('a_1', sec.a_1)
#    print('a', sec.a)
#    print('q', sec.q)
#    print('fM', sec.fM)
#    print('log_g_1', sec.log_g_1)
#    print('log_g_2', sec.log_g_2)
#    print('rho_1', sec.rho_1)
##    print('m_1', sec.m_1)
##    print('m_2', sec.m_2)

