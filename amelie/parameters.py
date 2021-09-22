#! /usr/bin/env python

from __future__ import (division, print_function, absolute_import)

__all__ = ['Parameter', 'OrbitalParameters', '_parameter_names']
#           '_allowed_set_parameters', '_allowed_lc_priors',
#           '_allowed_rv_priors']

import numpy as np
from scipy.stats import mode as scipy_mode
from astropy.constants import R_sun, M_sun, G, R_jup, M_jup, R_earth, M_earth
from conversions import *
from ellc import lc, rv

# (1) needs default parameters
# (2) decide on which parameters to use by default, suggest k and rsum
# (3) 


# NOTE: VEDAD!!! ellc.rv with RM now returns pri and sec

_parameter_names = {'allowed_set_parameters' : ('P',
                                                'Tpri',
                                                'rsum',
                                                'k',
                                                'cosi',
                                                'J',
                                                'f_s',
                                                'f_c',
                                                'Kpri',
                                                'light_3',
                                                'vsini_1',
                                                'lambda_1',
                                                'dPdt'),

                    'allowed_lc_priors' :      ('P',
                                                'Tpri',
                                                'Tsec',
                                                'rsum',
                                                'k',
                                                'cosi',
                                                'J',
                                                'f_s',
                                                'f_c',
                                                'D',
                                                'incl',
                                                'b',
                                                'bsec',
                                                'W',
                                                'Wsec',
                                                'r_1',
                                                'r_2',
                                                'w',
                                                'e',
                                                'light_3',
                                                'R_1',
                                                'R_2',
                                                'dPdt'),

                    'allowed_rv_priors' :      ('P',
                                                'Tpri',
                                                'cosi',
                                                'f_s',
                                                'f_c',
                                                'Kpri',
                                                'incl',
                                                'e',
                                                'w',
                                                'a',
                                                'a_1',
                                                'a_2',
                                                'vsini_1',
                                                'lambda_1',
                                                'q',
                                                'fM',
                                                'M_1',
                                                'M_2',
                                                'Mjup_1',
                                                'Mjup_2',
                                                'Mearth_2',
                                                'dPdt')}
#                                                'Mearth_2')}

#_allowed_set_parameters = ('P',
#                           'Tpri',
#                           'rsum',
#                           'k',
#                           'cosi',
#                           'J',
#                           'f_s',
#                           'f_c',
#                           'Kpri')
#
#
## allowed priors for LC fitting
#_allowed_lc_priors = ('P',
#                      'Tpri',
#                      'Tsec',
#                      'rsum',
#                      'k',
#                      'cosi',
#                      'J',
#                      'f_s',
#                      'f_c',
#                      'D',
#                      'incl',
#                      'r_1',
#                      'r_2',
#                      'w',
#                      'e')
#
## allowed priors for RV fitting
#_allowed_rv_priors = ('P',
#                      'Tpri',
#                      'cosi',
#                      'f_s',
#                      'f_c',
#                      'Kpri',
#                      'incl',
#                      'e',
#                      'w',
#                      'a_1',
#                      'a_2',
#                      'q',
#                      'fM',
#                      'M_1',
#                      'M_2',
#                      'Mjup_1',
#                      'Mjup_2',
#                      'Mearth_1',
#                      'Mearth_2')
class LimbDark(object):
    def __init__(self):#, channel):#, ld):

#        self.channel = channel
#        self.fname = ':'.join(['channel', channel])
#        self.ld = ld

#        if self.ld == 'lin':
#            pn = ('ldc_1',)
#        elif self.ld == 'quad' or self.ld == 'power-2':
#            pn = ('ldc_1', 'ldc_2')

#        self.parameter_names = pn + ('gdc_1', 'gdc_2', 'teff', 'logg', 'z')
        self.parameter_names = ('gdc_1', 'gdc_2', 'teff', 'logg', 'z')


        for name in self.parameter_names:
            setattr(self, '_'.join(['', name]), Parameter(name=name))


    def set_parameter(self, name, **kwargs):
        for key, val in kwargs.items():
            if key == 'value':
                key = '_'.join(['', key])
            setattr(getattr(self, '_'.join(['', name])), key, val)

    def _get_model(self, func, t, companion,
                   ld=None, ldc=None,
                   oversample=1, t_exp=None, in_transit=False, 
                   func_kwargs=None):
        """ get the specified model (ellc.lc() or ellc.rv())
            for the specified companion. Common function because most parameters
            overlap between the two functions """

        c = companion

        grid = c._warp_times(t)

        n_int = c._get_n_int(grid, oversample=oversample, in_transit=in_transit,
                             t_exp=t_exp)

        ellc_kwargs  = {'radius_1' : c.r_1.value,
                        'radius_2' : c.r_2.value,
                        'incl'     : c.incl.value,
                        'sbratio'  : c.J.value,
                        'period'   : c.P.value,
                        't_zero'   : c.Tpri.value,
                        'q'        : c.q.value,
                        'f_s'      : c.f_s.value,
                        'f_c'      : c.f_c.value,
                        'ld_1'     : self.ld,
                        'ldc_1'    : self.ldc,
                        'gdc_1'    : self.gdc.value,
                        't_exp'    : t_exp, 
                        'n_int'    : n_int,
                        'grid_1'   : 'very_sparse',
                        'grid_2'   : 'very_sparse',
                        'shape_1'  : "sphere"}

#        print('ld', self.ld)
#        print('ldc', self.ldc)

        ellc_kwargs.update(func_kwargs)

        return func(grid, **ellc_kwargs)

    def get_light_curve(self, t, companions, **kwargs):

#        if isinstance(companion, Companion):
#            companion = [companion]

        mod = np.ones_like(t)
        for c in companions:
            func_kwargs = {'func_kwargs': {'light_3' : c.light_3.value}}

#            all_kwargs  = {'func_kwargs': {'light_3' : c.light_3.value}}
#            all_kwargs.update(kwargs)
            kwargs.update(func_kwargs)
            
            mod *= self._get_model(lc, t, c,# func_kwargs=func_kwargs,
#                                  **all_kwargs)
                                  **kwargs)
#        print('mod', mod)

        return mod

    def get_rossiter_mclaughlin(self, t, companions, keplerian=None, **kwargs):
#        mod = np.zeros_like(t)
        mod = np.zeros((2, len(t)))
        for c in companions:
            func_kwargs  = {'func_kwargs': {'a'             : c.a.value,
                                            'q'             : c.q.value,
                                            'vsini_1'       : c.vsini_1.value,
                                            'lambda_1'      : c.lambda_1.value,
                                            'flux_weighted' : True}}
            kwargs.update(func_kwargs)
            # NOTE: VEDAD!!! ellc.rv with RM now returns pri and sec
            mod += np.array(self._get_model(rv, t, c, **kwargs))# - 
#                    self.get_radial_velocity(t, c, **kwargs)
#                    )
            
        if keplerian is None:
#            keplerian = np.zeros_like(t)
            keplerian = np.zeros((2, len(t)))
        elif keplerian.shape[0] < 2:
            tmp = np.zeros_like(t)
            keplerian = np.vstack((keplerian, tmp))

        return mod - keplerian


#    def get_radial_velocity(self, t, companions, **kwargs):
#        # ellc.rv() parameters
#
#        mod = np.zeros_like(t)
#        for c in companions:
##            func_kwargs = {'func_kwargs' : {}}
#            func_kwargs  = {'func_kwargs': {'a'             : c.a.value,
#                                            'q'             : c.q.value,
#            kwargs.update(func_kwargs)
#            mod += self.get_model(ellc.rv, t, c, **kwargs)[0]
#        
##        all_kwargs.update(kwargs)
#
#        return mod

#    @property
#    def ldc_1(self):
#        return self._ldc_1
#
#    @property
#    def ldc_2(self):
#        return self._ldc_2
#
#    @property
#    def ldc(self):
#        return [self.ldc_1.value, self.ldc_2.value]

#class Deterministic(Parameter):
#    def __init__(self, name, expr):
#        super(Deterministic, self).__init__(*args, **kwargs)
#        
#        self.expr = expr
#
#    def value(self, *args):
#        return self.expr(*args)
class Power2LimbDark(LimbDark):
    def __init__(self, u=None, gdc=None):

        super(Power2LimbDark, self).__init__()

        self.ld = 'power-2'

        self.parameter_names += ('ldc_1' , 'ldc_2',
                                 'ldc_1_maxted', 'ldc_2_maxted')

        for name in self.parameter_names:
            setattr(self, '_'.join(['', name]), Parameter(name=name))

        # always set u1, u2 (incl priors if needed)
        # convert u1, u2 (and errors) to q1, q1
        # when new u1, u2 is deterministic on q1, q2
        # when checking for ld error, add q1, q2 to free params instead

        if u is None:
            u = [None, None]

        self.set_parameter('ldc_1', value=u[0])#, bounds=(-1, 1))
        self.set_parameter('ldc_2', value=u[1])#, bounds=(-1, 1))

        self.set_parameter('ldc_1_maxted', value=None)#, bounds=(0, 1))
        self.set_parameter('ldc_2_maxted', value=None)#, bounds=(0, 1))

        self.set_parameter('gdc_1', value=gdc)

#    def set_parameter(self, name, **kwargs):
#        for key, val in kwargs.items():
#            if key == 'value':
#                key = '_'.join(['', key])
#            setattr(getattr(self, '_'.join(['', name])), key, val)


   


#    def to_kipping(self, ldc):
#        q1 = (self.ldc_1 + self.ldc_2)**2
#        q2 = self.ldc_1 / (2 * (self.ldc_1 + self.ldc_2))
#        return [q1, q2]
    @property
    def gdc(self):
        return self._gdc_1
#
    @property
    def ldc(self):
        return [self.ldc_1.value, self.ldc_2.value]

    @property
    def ldc_1_maxted(self):
        return self._ldc_1_maxted
#        return (self.ldc_1 + self.ldc_2)**2

    @property
    def ldc_2_maxted(self):
        return self._ldc_2_maxted
#        return 

    @property
    def ldc_1(self):
        if None in [self.ldc_1_maxted.value, self.ldc_2_maxted.value]:
            return self._ldc_1
        v = 1 - self.ldc_1_maxted.value + self.ldc_2_maxted.value
        self.set_parameter('ldc_1', value=v)
#        return 2 * np.sqrt(self.ldc_1_maxted) * self.ldc_2_maxted
        return self._ldc_1

    @property
    def ldc_2(self):
        if None in [self.ldc_1_maxted.value, self.ldc_2_maxted.value]:
            return self._ldc_2
        v = np.log2((1 - self.ldc_1_maxted.value + self.ldc_2_maxted.value)/
                self.ldc_2_maxted.value)
        self.set_parameter('ldc_2', value=v)
#        return self._ldc_2
        return self._ldc_2


class QuadLimbDark(LimbDark):
    def __init__(self, u=None, gdc=None):

        super(QuadLimbDark, self).__init__()

        self.ld = 'quad'

        self.parameter_names += ('ldc_1' , 'ldc_2',
                                 'ldc_1_kipping', 'ldc_2_kipping')

        for name in self.parameter_names:
            setattr(self, '_'.join(['', name]), Parameter(name=name))

        # always set u1, u2 (incl priors if needed)
        # convert u1, u2 (and errors) to q1, q1
        # when new u1, u2 is deterministic on q1, q2
        # when checking for ld error, add q1, q2 to free params instead

        if u is None:
            u = [None, None]

        self.set_parameter('ldc_1', value=u[0])#, bounds=(-1, 1))
        self.set_parameter('ldc_2', value=u[1])#, bounds=(-1, 1))

        self.set_parameter('ldc_1_kipping', value=None)#, bounds=(0, 1))
        self.set_parameter('ldc_2_kipping', value=None)#, bounds=(0, 1))

        self.set_parameter('gdc_1', value=gdc)

#    def set_parameter(self, name, **kwargs):
#        for key, val in kwargs.items():
#            if key == 'value':
#                key = '_'.join(['', key])
#            setattr(getattr(self, '_'.join(['', name])), key, val)


   


#    def to_kipping(self, ldc):
#        q1 = (self.ldc_1 + self.ldc_2)**2
#        q2 = self.ldc_1 / (2 * (self.ldc_1 + self.ldc_2))
#        return [q1, q2]
#
    @property
    def gdc(self):
        return self._gdc_1

    @property
    def ldc(self):
        return [self.ldc_1.value, self.ldc_2.value]

    @property
    def ldc_1_kipping(self):
        return self._ldc_1_kipping
#        return (self.ldc_1 + self.ldc_2)**2

    @property
    def ldc_2_kipping(self):
        return self._ldc_2_kipping
#        return 

    @property
    def ldc_1(self):
        if None in [self.ldc_1_kipping.value, self.ldc_2_kipping.value]:
            return self._ldc_1
        v = 2 * np.sqrt(self.ldc_1_kipping.value) * self.ldc_2_kipping.value
        self.set_parameter('ldc_1', value=v)
        return 2 * np.sqrt(self.ldc_1_kipping) * self.ldc_2_kipping

    @property
    def ldc_2(self):
        if None in [self.ldc_1_kipping.value, self.ldc_2_kipping.value]:
            return self._ldc_2
        v = (np.sqrt(self.ldc_1_kipping.value) * 
                (1 - 2 * self.ldc_2_kipping.value))
        self.set_parameter('ldc_2', value=v)
        return self._ldc_2

#    @property
#    def ldc_det(self):
#        return [self.ldc_1_det.value, self.ldc_2_det.value]



class Parameter(object):
    def __init__(self, name, value=None, error=None, bounds=None,
                             mu=None, sd=None):

        self.name   = name
        self._value = value
        self.error  = error
        self.bounds = bounds
        self.mu     = mu
        self.sd     = sd

#        super(Parameter, self).__init__()

    @property
    def value(self):
#        print('calling value for {:}'.format(self.name))
        return self._value

#    @property
#    def _has_prior(self):
#        if ((self.bounds is not None) or 
#            ((self.mu is not None) and (self.sd is not None))):
#            return True
#        return False

    @property
    def log_prior(self):
#        if self._has_prior:
        l = 0
        if self.bounds is None:
#            print(self.name, ': should see this a lot')
            l += 0
        elif not ((self.value > self.bounds[0]) and (self.value < self.bounds[1])):
#            print('should not see this')
            return -np.inf
        else:
            pass

        if (self.mu is not None) and (self.sd is not None):
#            print('should not see this')
            l += -0.5 * ((self.value - self.mu) / self.sd)**2

        return l 

    @property
    def std(self):
        return np.std(self.value)

    @property
    def mode(self):
        return scipy_mode(self.value, nan_policy='omit', axis=None).mode[0]

    @property
    def median(self):
        return np.nanmedian(self.value)

    @property
    def mean(self):
        return np.nanmean(self.value)

    def percentile(self, value):
        return np.percentile(self.value, value)

    @property
    def std_lower(self):
        return np.percentile(self.value, 50) - np.percentile(self.value, 16)

    @property
    def std_upper(self):
        return  np.percentile(self.value, 84) - np.percentile(self.value, 50)

#class OrbitalParameters(object):
#    def __init__(self):
#        
#        # these parameters are allowed to vary
#        self.varied_parameters = ('P', 'Tpri',
#                    'rsum', 'k', 'cosi', 'J',
#                             'f_s', 'f_c', 'Kpri',
##                             'vsini_1', 'lambda_1'
#                             )
#
#        # these parameter values are derived from the varied parameters
#        self.derived_parameters = ('Tsec', 'D', 'incl', 'r_1', 'r_2', 'w', 'e',
#                             'a_1', 'a', 'a_2', 'q', 'fM', 'R_1', 'R_2',
#                             'Rjup_1', 'Rjup_2', 'Rearth_1', 'Rearth_2',
#                             'M_1', 'M_2', 'Mjup_1', 'Mjup_2',
#                             'Mearth_1', 'Mearth_2')
#
#        self.parameter_names = self.varied_parameters + self.derived_parameters
#
#        for name in self.parameter_names:
#            print('creating ', name)
#            setattr(self, '_'.join(['', name]), Parameter(name=name))
#
#        # add default numerical values for given parameters
##        has_default_values = ('P', 'Tpri', 'cosi', 'J', 'f_s', 'f_c', 'q')
#
#        self.set_parameter('P',    value=1)
#        self.set_parameter('Tpri', value=0)
#        self.set_parameter('cosi', value=0)
#        self.set_parameter('J',    value=0)
#        self.set_parameter('f_s',  value=0)
#        self.set_parameter('f_c',  value=0)
#        self.set_parameter('q',    value=1)
#
#
##        super(OrbitalParameters, self).__init__()
#
#
#    def set_parameter(self, name, **kwargs):
#        for key, val in kwargs.items():
#            if key == 'value':
#                key = '_'.join(['', key])
#            setattr(getattr(self, '_'.join(['', name])), key, val)
#
#
##    def _has_priors(self):
##        return tuple([p for p in self.parameter_names
##                        if getattr(getattr(self, p), 'bounds') is not None
##                        and if getattr(getattr(self, p), ''
#                      
##        setattr(self, getattr(super(Companion, self).parameter, '_value'),
##                Parameter(parameter, value=value, error=error,
##                          bounds=bounds, gprior=gprior))
##        setattr(self, parameter,
##                Parameter(parameter, value=value, error=error,
##                          bounds=bounds, gprior=gprior))
#
##    def get_parameter_names(self):
##        return (p
##                for p in self.__dict__.keys()
##                if p.startswith('_'))
##
#    # impact parameter
#    @property
#    def b(self):
#        v = get_impact_parameter(self.cosi, self.r_1, self.e, self.w)
#        self.set_parameter('b', value=v)
#        return self._b
#
#    # orbital period
#    @property
#    def P(self):
#        return self._P
#
#    # radius ratio, r_2/r_1
#    @property
#    def k(self):
#        return self._k
#
#    # surface brightness ratio, J_2/J_1
#    @property
#    def J(self):
#        return self._J
#    
#    # transit depth
#    @property
#    def D(self):
#        try:
#            v = get_transit_depth(self.k.value, self.J.value)
#        except TypeError:
#            v = None
#        self.set_parameter('D', value=v)
#        return self._D
#
#    # orbital inclination
#    @property
#    def incl(self):
#        v = get_inclination(self.cosi.value)
#        self.set_parameter('incl', value=v)
#        return self._incl
#
#    # transit duration (width)
#    @property
#    def W(self):
#        try:
#            v = get_transit_width(self.r_1.value, self.P.value, self.k.value,
#                                  self.b.value, self.e.value, self.w.value)
#        except TypeError:
#            self.set_parameter('W', value=v)
#        return self._W
#
#    # cosine of orbital inclination
#    @property
#    def cosi(self):
#        return self._cosi
#
#
#    # scaled primary radius, R_1/a
#    @property
#    def r_1(self):
#        try:
#            v = get_radius_scaled_primary(self.rsum.value, self.k.value)
#        except TypeError:
#            v = None
#        self.set_parameter('r_1', value=v)
#        return self._r_1
#
#    # scaled secondary radius
#    @property
#    def r_2(self):
#        v = get_radius_scaled_secondary(self.r_1.value, self.k.value)
#        self.set_parameter('r_2', value=v)
#        return self._r_2
#
#    # scaled sum of radii
#    @property
#    def rsum(self):
#        return self._rsum
#
#    # time of primary eclipse
#    @property
#    def Tpri(self):
##        if self._Tsec.value is not None:
##            v = get_time_of_primary_eclipse(self.Tsec.value, self.e.value,
##                                            self.w.value, self.P.value,
##                                            self.incl.value)
##            self.set_parameter('Tpri', value=v)
#        return self._Tpri
#
##        if self._Tpri.value is None and self._Tsec.value is not None:
##            v = get_time_of_primary_eclipse(self.Tsec.value, self.e.value,
##                                            self.w.value, self.P.value,
##                                            self.incl.value)
##            self.set_parameter('Tpri', value=v)
##        return self._Tpri
#
#    # time of secondary eclipse
#    @property
#    def Tsec(self):
##        if self._Tsec.value is None and self._Tpri.value is not None:
#        v = get_time_of_secondary_eclipse(self.Tpri.value, self.e.value,
#                                          self.w.value, self.P.value,
#                                          self.incl.value)
#        self.set_parameter('Tsec', value=v)
#        return self._Tsec
#
#    @property
#    def P(self):
#        return self._P
#
#    # semi-major axis of binary
#    @property
#    def a(self):
#        try:
#            v = get_semimajor_axis(self.a_1.value, self.q.value)
#        except TypeError:
#            v = None
#        self.set_parameter('a', value=v)
#        return self._a
#
#    # barycentric semi-major axis of primary
#    @property
#    def a_1(self):
#        try:
#            v = get_semimajor_axis_primary(self.e.value, self.incl.value,
#                                           self.Kpri.value, self.P.value)
#        except TypeError:
#            v = None
#        self.set_parameter('a_1', value=v)
#        return self._a_1
#
#    @property
#    def a_2(self):
#        try:
#            v = get_semimajor_axis_secondary(self.a.value, self.a_1.value)
#        except TypeError:
#            v = None
#        self.set_parameter('a_2', value=v)
#        return self._a_2
#
#    # eccentricity
#    @property
#    def e(self):
#        v = get_eccentricity(self.f_s.value, self.f_c.value)
#        self.set_parameter('e', value=v)
#        return self._e
#
#    # mass function
#    @property
#    def fM(self):
#        try:
#            v = get_mass_function(self.e.value, self.Kpri.value, self.P.value)
#        except TypeError:
#            v = None
#        self.set_parameter('fM', value=v)
#        return self._fM
#
#    # radial component of eccentricity, sqrt(e).sin(w)
#    @property
#    def f_s(self):
#        return self._f_s
#
#    # tangential component of eccentricity, sqrt(e).cos(w)
#    @property
#    def f_c(self):
#        return self._f_c
#
#    # radial velocity semi-amplitude
#    @property
#    def Kpri(self):
#        return self._Kpri
#
#    # log surface gravity of primary
#    @property
#    def log_g_1(self):
#        v = get_surface_gravity_primary(self.fM.value, self.P.value,
#                                        self.r_1.value, self.incl.value,
#                                        self.q.value)
#        self.set_parameter('log_g_1', value=v)
#        return self._log_g_1
#
#    # log surface gravity of secondary
#    @property
#    def log_g_2(self):
#        v = get_surface_gravity_secondary(self.fM.value, self.P.value,
#                                          self.r_2.value, self.incl.value)
#        self.set_parameter('log_g_2', value=v)
#        return self._log_g_2
#
#    # mass of primary (solar units)
#    @property
#    def M_1(self):
#        return self._M_1
#
#    # mass of secondary (solar units)
#    @property
#    def M_2(self):
#        if self.M_1.value is None:
#            raise ValueError("""M_1 needs to be set to a value before M_2 can be
#                             computed, e.g. `self._M_1 = 0.8`.
#                             Warning: the mass calculation assumes M_2 << M_1,
#                             which may be a bad approximation since you are
#                             modelling a (single line) binary""")
#        # assuming M_2 << M_1
#        v = get_mass_secondary_solar(self.M_1.value, self.Kpri.value,
#                                     self.e.value, self.incl.value,
#                                     self.P.value)
#        self.set_parameter('M_2', value=v)
#        return self._M_2
#
#    # mass of primary (jovian units)
#    @property
#    def Mjup_1(self):
#        v = get_mass_primary_jupiter(self.M_1.value)
#        self.set_parameter('Mjup_1', value=v)
#        return self._Mjup_1
#
#    # mass of secondary (jovian units)
#    @property
#    def Mjup_2(self):
#        v = get_mass_secondary_jupiter(self.M_2.value)
#        self.set_parameter('Mjup_2', value=v)
#        return self._Mjup_2
#
#    # mass of secondary (earth units)
#    @property
#    def Mearth_2(self):
#        v = get_mass_secondary_earth(self.M_2.value)
#        self.set_parameter('Mearth_2', value=v)
#        return self._Mearth_2
#
#    # mass ratio, M_2/M_1
#    @property
#    def q(self):
#        if self.M_1.value is None:
#            v = self._q.value
#        else:
#            v = get_mass_ratio(self.M_2.value, self.M_1.value)
#        self.set_parameter('q', value=v)
#        return self._q
#
#    # density of primary
#    @property
#    def rho_1(self):
#        """ uses M_2 << M_1 approximation, may be bad depending on the case """
#        v = get_density_primary(self.M_1.value, self.q.value, self.P.value,
#                                self.r_1.value)
#        self.set_parameter('rho_1', value=v)
#        return self._rho_1
#
#    @property
#    def rho_2(self):
#        if self.M_1.value is None:
#            raise ValueError("""M_1 needs to be set to a value before rho_2 can be
#                             computed, e.g. `self._M_1 = 0.8`.
#                             Warning: the calculation assumes M_2 << M_1,
#                             which may be a bad approximation""")
#        v = get_density_secondary(self.M_2.value, self.R_2.value)
#        self.set_parameter('rho_2', value=v)
#        return self._rho_2
#
#    # primary radius (solar units)
#    @property
#    def R_1(self):
#        v = get_radius_primary_solar(self.r_1.value, self.a.value)
#        self.set_parameter('R_1', value=v)
#        return self._R_1
#
#    # secondary radius (solar units)
#    @property
#    def R_2(self):
#        v = get_radius_secondary_solar(self.r_2.value, self.a.value)
#        self.set_parameter('R_2', value=v)
#        return self._R_2
#
#    # primary radius (jovian units)
#    @property
#    def Rjup_1(self):
#        v = get_radius_primary_jupiter(self.R_1.value)
#        self.set_parameter('Rjup_1', value=v)
#        return self._Rjup_1
#
#    # secondary radius (jovian units)
#    @property
#    def Rjup_2(self):
#        v = get_radius_secondary_jupiter(self.R_2.value)
#        self.set_parameter('Rjup_2', value=v)
#        return self._Rjup_2
#
#    # secondary radius (earth units)
#    @property
#    def Rearth_2(self):
#        v = get_radius_secondary_earth(self.R_2.value)
#        self.set_parameter('Rearth_2', value=v)
#        return self._Rearth_2
#
#    # argument of periastron
#    @property
#    def w(self):
#        v = get_argument_of_periastron(self.f_s.value, self.f_c.value)
##        v = np.rad2deg(np.arctan2(self.f_s.value, self.f_c.value))
#        self.set_parameter('w', value=v)
#        return self._w

        

# companion = amelie.Planet(name='pimen-c')
# -> creates OrbitalParams object for companion: par = OrbitalParams()
# set_parameter:
# par._P = Parameter(*args, **kwargs)
# companion.P = Parameter(
#class OrbitalParameters(object):
#    def __init__(self):
#        
##        super(OrbitalParams, self).__init__()
#
#        # ephemerides
#        self._P    = None
#        self._Tpri = None
#        self._Tsec = None
#
#        # eclipse parameters
#        self._rsum = None
#        self._k    = None
#        self._cosi = None
#        self._J    = None
#
#        # RV parameters
#        self._f_s  = 0
#        self._f_c  = 0
#        self._Kpri = None
#        self._M_1  = None
#        self._q    = 1
#
#        # RM parameters
#        self._vsini_1  = None
#        self._lambda_1 = None
#
##        self.flux_weighted = False
#
#    def get_parameter_names(self):
#        return (p
#                for p in self.__dict__.keys()
#                if p.startswith('_'))
#
#    def _delta_t(self):
#        """
#        time between primary and secondary eclipse, from Hilditch 2001
#        """
#        c = self.e / np.sqrt(1 - self.e**2) * np.cos(np.deg2rad(self.w))
#        return self.P / (2*np.pi) * (np.pi + 2*np.arctan(c) +
#               np.sin(2*np.arctan(c)) + 2*self.e*np.cos(np.deg2rad(self.w)) / 
#               np.tan(np.deg2rad(self.incl))**2)
#
#    @property
#    def b(self):
#        return self.cosi / self.r_1 * ((1 - self.e**2) /
#                                       (1 + self.e * np.sin(self.w)))
#
#    # transit depth
#    @property
#    def D(self):
#        return self.k**2 * (1 - self.s)
#
#    # orbital inclination
#    @property
#    def incl(self):
#        return np.rad2deg(np.arccos(self.cosi))
#       
#    # transit duration (width)
#    @property
#    def W(self):
#        return (self.r_1 * self.P * 
#                np.sqrt((1 - self.k)**2 - self.b**2) / np.pi *
#                np.sqrt(1 - self.e**2) / (1 - self.e * np.deg2rad(self.w)))
#
#
#    # cosine of orbital inclination
#    @property
#    def cosi(self):
#        return self._cosi.value
#
#    # radius ratio
#    @property
#    def k(self):
#        return self._k.value
#
#    # scaled primary radius
#    @property
#    def r_1(self):
#        return self.rsum / (1 + self.k)
#
#    # scaled secondary radius
#    @property
#    def r_2(self):
#        return self.r_1 * self.k
#
#    # scaled sum of radii
#    @property
#    def rsum(self):
#        return self._rsum.value
#
#    # time of primary eclipse
#    @property
#    def Tpri(self):
#        if self._Tpri.value is None and self._Tsec.value is not None:
#            return self.Tsec - self._delta_t()
#        return self._Tpri.value
#
#    # time of secondary eclipse
#    @property
#    def Tsec(self):
#        if self._Tsec is None and self._Tpri.value is not None:
#            return self.Tpri + self._delta_t()
#        return self._Tsec.value
#
#    @property
#    def P(self):
#        return self._P.value
#
#    # semi-major axis of binary
#    @property
#    def a(self):
#        return self.a_1 * (1 + 1/self.q)
#
#    # barycentric semi-major axis of primary
#    @property
#    def a_1(self):
#        days2seconds = 24*60*60
#        km2m         = 1e3
#        return (np.sqrt(1 - self.e**2) / (2*np.pi * np.sin(np.deg2rad(self.incl))) *
#                self.Kpri * km2m * (self.P * days2seconds) / R_sun.value)
#
#    @property
#    def a_2(self):
#        return self.a - self.a_1
#
#    # eccentricity
#    @property
#    def e(self):
#        return self.f_s**2 + self.f_c**2
#
#    # mass function
#    @property
#    def fM(self):
#        # period in units of days, K in units of km/s
##        return (1.0361e-7) * (1 - self.e**2)**1.5 * (self.Kpri/1000)**3 * self.P
#        return (1.0361e-7) * (1 - self.e**2)**1.5 * self.Kpri**3 * self.P
#
#    # radial component of eccentricity, sqrt(e).sin(w)
#    @property
#    def f_s(self):
#        return self._f_s.value
#
#    # tangential component of eccentricity, sqrt(e).cos(w)
#    @property
#    def f_c(self):
#        return self._f_c.value
#
#    # radial velocity semi-amplitude
#    @property
#    def Kpri(self):
#        return self._Kpri.value
#
#    # log surface gravity of primary
#    @property
#    def log_g_1(self):
#        return (3.18987 + (np.log10(self.fM) - 4*np.log10(self.P))/3 -
#                np.log10(self.r_1**2 * np.sin(np.deg2rad(self.incl))) -
#                np.log10(self.q))
#
#    # log surface gravity of secondary
#    @property
#    def log_g_2(self):
#        return (3.18987 + (np.log10(self.fM) - 4*np.log10(self.P))/3 -
#                np.log10(self.r_2**2 * np.sin(np.deg2rad(self.incl))))
#
#    # mass of primary (solar units)
#    @property
#    def M_1(self):
#        return self._M_1.value
#
#    # mass of secondary (solar units)
#    @property
#    def M_2(self):
#        if self.M_1 is None:
#            raise ValueError("""M_1 needs to be set to a value before M_2 can be
#                             computed, e.g. `self._M_1 = 0.8`.
#                             Warning: the mass calculation assumes M_2 << M_1,
#                             which may be a bad approximation since you are
#                             modelling a (single line) binary""")
#        # assuming M_2 << M_1
#        d2s  = 24*60*60 # convert days to seconds
#        km2m = 1e3
#        return ((self.M_1 * M_sun.value)**(2/3) * self.Kpri * km2m * np.sqrt(1 - self.e**2) /
#                np.sin(np.deg2rad(self.incl)) * 
#                (self.P * d2s / (2*np.pi * G.value))**(1/3) / M_sun.value)
#
#    # mass of primary (jovian units)
#    @property
#    def Mjup_1(self):
#        return self._M_1.value * M_sun.value / M_jup.value
#
#    # mass of secondary (jovian units)
#    @property
#    def Mjup_2(self):
#        return self.M_2 * M_sun.value / M_jup.value
#
#    # mass of secondary (earth units)
#    @property
#    def Mearth_2(self):
#        return self.M_2 * M_sun.value / M_earth.value
#
#    # orbital period
#    @property
#    def P(self):
#        return self._P.value
#
#    # mass ratio, M_2/M_1
#    @property
#    def q(self):
#        if self.M_1 is None:
#            return self._q
#        return self.M_2 / self.M_1
#
#    # density of primary
#    @property
#    def rho_1(self):
#        """ uses M_2 << M_1 approximation, may be bad depending on the case """
#        si2cgs = 1000. / (100**3)
#        days2seconds = 24*60*60
#        f = 1 if self.M_1 is None else (1 + self.q)
#        return (3 * np.pi / (G.value * (self.P * days2seconds)**2 * f) /
#                self.r_1**3) * si2cgs
#
#    @property
#    def rho_2(self):
#        if self.M_1 is None:
#            raise ValueError("""M_1 needs to be set to a value before rho_2 can be
#                             computed, e.g. `self._M_1 = 0.8`.
#                             Warning: the calculation assumes M_2 << M_1,
#                             which may be a bad approximation""")
#        si2cgs = 1000. / (100**3)
##        return self.M_2*M_sun.value / (self.R_2*R_sun.value)**3 * si2cgs
#        return 3 * self.M_2*M_sun.value / (4 * np.pi * (self.R_2*R_sun.value)**3) * si2cgs
#
#    # primary radius (solar units)
#    @property
#    def R_1(self):
#        return self.r_1 * self.a
#
#    # secondary radius (solar units)
#    @property
#    def R_2(self):
#        return self.r_2 * self.a
#
#    # primary radius (jovian units)
#    @property
#    def Rjup_1(self):
#        return self.R_1 * R_sun.value / R_jup.value
#
#    # secondary radius (jovian units)
#    @property
#    def Rjup_2(self):
#        return self.R_2 * R_sun.value / R_jup.value
#
#    # secondary radius (earth units)
#    @property
#    def Rearth_2(self):
#        return self.R_2 * R_sun.value / R_earth.value
#
#    # surface brightness ratio
#    @property
#    def J(self):
#        return self._J.value
#
##    # time of primary eclipse
##    @property
##    def Tpri(self):
##        return self._Tpri.value
##
##    # time of secondary eclipse
##    @property
##    def Tsec(self):
##        if self._Tsec is None and self._Tpri.value is not None:
##            return self.Tpri + self._delta_t()
##        return self._Tsec
#
#    # argument of periastron
#    @property
#    def w(self):
#        return np.rad2deg(np.arctan2(self.f_s, self.f_c))

