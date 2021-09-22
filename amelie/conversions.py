#!/usr/bin/env python

from __future__ import (print_function, division)

__all__ = ['get_impact_parameter', 'get_eclipse_impact_parameter', 
           'get_transit_depth', 'get_inclination',
           'get_transit_width', 'get_eclipse_width',
           'get_radius_scaled_primary',
           'get_radius_scaled_secondary', 'get_time_of_primary_eclipse',
           'get_time_of_secondary_eclipse', 'get_semimajor_axis',
           'get_semimajor_axis_primary', 'get_semimajor_axis_secondary',
           'get_eccentricity', 'get_argument_of_periastron',
           'get_mass_function', 'get_surface_gravity_primary',
           'get_surface_gravity_secondary',
           'get_mass_secondary_solar', 'get_mass_primary_jupiter', 
           'get_mass_secondary_jupiter', 'get_mass_secondary_earth',
           'get_mass_ratio', 'get_density_primary',
           'get_density_secondary', 'get_radius_primary_solar',
           'get_radius_secondary_solar', 'get_radius_primary_jupiter',
           'get_radius_secondary_jupiter', 'get_radius_secondary_earth',
           '_check_valid']

import numpy as np
from astropy.constants import R_sun, M_sun, G, R_jup, M_jup, R_earth, M_earth

def get_delta_t_primary_secondary_eclipse(e, w, P, incl):
    """
    time between primary and secondary eclipse, from Hilditch 2001
    """
    c = (e/ np.sqrt(1 - e**2) * np.cos(np.deg2rad(w)))
    return P / (2*np.pi) * (np.pi + 2*np.arctan(c) + np.sin(2*np.arctan(c)) +
           2*e*np.cos(np.deg2rad(w)) /  np.tan(np.deg2rad(incl))**2)

def _check_valid(func, *args):
    try:
        return func(*args)
    except TypeError:
        return None

def get_impact_parameter(cosi, r_1, e, w):
    return cosi / r_1 * ((1 - e**2) / (1 + e * np.sin(w)))

def get_eclipse_impact_parameter(cosi, r_1, e, w):
    return cosi / r_1 * ((1 - e**2) / (1 - e * np.sin(w)))

def get_transit_depth(k, J):
    return k**2 * (1 - J)

def get_inclination(cosi):
    return np.rad2deg(np.arccos(cosi))

def get_eclipse_width(r_1, P, k, b, e, w, incl):
    return  (P / np.pi * np.arcsin(r_1 * np.sqrt((1 + k)**2 - b**2) / 
             np.sin(np.deg2rad(incl))) *
             np.sqrt(1 - e**2) / (1 - e*np.sin(np.deg2rad(w))))

def get_transit_width(r_1, P, k, b, e, w, incl):
    """ THIS MIGHT NOT BE CORRECT """
#    return  (r_1 * P * np.sqrt((1 + k)**2 - b**2) / np.pi * 
#             np.sqrt(1 - e**2) / (1 + e*np.sin(np.deg2rad(w))))
    return  (P / np.pi * np.arcsin(r_1 * np.sqrt((1 + k)**2 - b**2) / 
             np.sin(np.deg2rad(incl))) *
             np.sqrt(1 - e**2) / (1 + e*np.sin(np.deg2rad(w))))

def get_radius_scaled_primary(rsum, k):
    return rsum / (1 + k)

def get_radius_scaled_secondary(r_1, k):
    return r_1 * k

def get_time_of_primary_eclipse(Tsec, e, w, P, incl):
    dt = get_delta_t_primary_secondary_eclipse(e, w, P, incl)
    return Tsec - dt

def get_time_of_secondary_eclipse(Tpri, e, w, P, incl):
    dt = get_delta_t_primary_secondary_eclipse(e, w, P, incl)
    return Tpri + dt

def get_semimajor_axis(a_1, q):
    return a_1 * (1 + 1/q)

def get_semimajor_axis_primary(e, incl, Kpri, P):
    d2s  = 24*60*60
    km2m = 1e3
    return (np.sqrt(1 - e**2) / (2*np.pi * np.sin(np.deg2rad(incl))) *
            Kpri * km2m * (P * d2s) / R_sun.value)

def get_semimajor_axis_secondary(a, a_1):
    return a - a_1

def get_eccentricity(f_s, f_c):
    return f_s**2 + f_c**2

def get_argument_of_periastron(f_s, f_c):
    return np.rad2deg(np.arctan2(f_s, f_c))

def get_mass_function(e, Kpri, P):
    return (1.0361e-7) * (1 - e**2)**1.5 * Kpri**3 * P

def get_surface_gravity_primary(fM, P, r_1, incl, q):
    return (3.18987 + (np.log10(fM) - 4*np.log10(P))/3 -
            np.log10(r_1**2 * np.sin(np.deg2rad(incl))) - np.log10(q))

def get_surface_gravity_secondary(fM, P, r_2, incl):
    return (3.18987 + (np.log10(fM) - 4*np.log10(P))/3 -
            np.log10(r_2**2 * np.sin(np.deg2rad(incl))))

def get_mass_secondary_solar(M_1, Kpri, e, incl, P):
    d2s  = 24*60*60 # convert days to seconds
    km2m = 1e3
    return ((M_1 * M_sun.value)**(2/3) * Kpri * km2m * 
            np.sqrt(1 - e**2) / np.sin(np.deg2rad(incl)) * 
           (P * d2s / (2*np.pi * G.value))**(1/3) / M_sun.value)

def get_mass_primary_jupiter(M_1):
    return M_1 * M_sun.value / M_jup.value

def get_mass_secondary_jupiter(M_2):
    return M_2 * M_sun.value / M_jup.value

def get_mass_secondary_earth(M_2):
    return M_2 * M_sun.value / M_earth.value

def get_mass_ratio(M_2, M_1):
    return M_2 / M_1

def get_density_primary(M_1, q, P, r_1):
    si2cgs = 1000. / (100**3)
    d2s = 24*60*60
    f = 1 if M_1 is None else (1 + q)
    return 3 * np.pi / (G.value * (P * d2s)**2 * f) / r_1**3 * si2cgs

def get_density_secondary(M_2, R_2):
    si2cgs = 1000. / (100**3)
#   return self.M_2*M_sun.value / (self.R_2*R_sun.value)**3 * si2cgs
    return 3 * M_2*M_sun.value / (4 * np.pi * (R_2*R_sun.value)**3) * si2cgs

def get_radius_primary_solar(r_1, a):
    return r_1 * a

def get_radius_secondary_solar(r_2, a):
    return r_2 * a

def get_radius_primary_jupiter(R_1):
    return R_1 * R_sun.value / R_jup.value

def get_radius_secondary_jupiter(R_2):
    return R_2 * R_sun.value / R_jup.value

def get_radius_secondary_earth(R_2):
    return R_2 * R_sun.value / R_earth.value



