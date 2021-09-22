#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import)

__all__ = ['Photometry', 'RadialVelocity', 'DataSet']

import os
import string
import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column, vstack
from parameters import Parameter

#class Photometry(Data):
#    def __init__(self, interpolate=False, **kwargs):
#        self.interpolate = interpolate
#
#        super(Photometry, self).__init__(**kwargs)

class Data(object):
    def __init__(self, x, y, filename=None,
#                 x=None, y=None,
                 y_err=None, t_exp=None, 
                 channel=None,
                 jitter=None, 
                 trend=None,
                 knot_spacing=None,
                 in_transit=False,
                 oversample=1,
#                 independent_variable_column=None,
                flux_weighted=False,
                 loadtxt_kwargs={}):
        """
        polynomial : list
                   : list that contains 2-tuples that represent the polynomial
                     order, and column number in `filename` of the independent
                     variable. For example, a second-order polynomial in time
                     would be `polynomial = [(2, 0)]` because time is the first
                     column in `filename`.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if y_err is not None:
            y_err = np.asarray(y_err, dtype=np.float64)

        self.channel    = channel
        self.data_file  = filename
        self.trend      = trend 
        self.knot_spacing = knot_spacing
        self._jitter     = jitter
        self.oversample = oversample
        self.in_transit = in_transit
        self.flux_weighted = flux_weighted

#        self.variable_column       = []
#        self.polynomial_models     = []

#        if polynomial is not None:
#            for p in polynomial:
#                self.polynomial_models.append(p[0])
#                self.variable_column.append(p[1])

        if self.trend is not None:
            self.slices = self._get_slices()

        if jitter is None:
#            self.jitter = Parameter(':'.join([channel, 'jitter']), value=0)
            self._jitter = Parameter("jitter", value=0)
#        else:
#            self.jitter.name = ":".join([channel, "jitter"])

#        if filename is not None:
#            # add additional np.loadtxt() kwargs
#            loadtxt_kwargs.update({'unpack'  : False,
#                                   'usecols' : (0,1,2,3)})
#            self.read_data_file(filename, **loadtxt_kwargs)
#        else:
        self.x     = x
        self.y     = y
        self.y_err = y_err
        self.t_exp = t_exp
        self.resid = np.zeros_like(self.y)

        self._allowed_priors = ("jitter",)

#    def read_data_file(self, filename, **kwargs):
#
#        data = np.loadtxt(filename, **kwargs)
#
#        if self.polynomial is not None:
#            kwargs['usecols'] = self.varcols
#            self.independent_variables = np.loadtxt(self.data_file, **kwargs)
#
#        self.x     = data[:,0]
#        self.y     = data[:,1]
#        self.y_err = data[:,2]
#        self.t_exp = data[:,3]
#        self.resid = np.zeros_like(self.y)
    @property
    def jitter(self):
        return self._jitter
            
    def set_parameter(self, name, **kwargs):
        for key, val in kwargs.items():
            if key == 'value':
                key = '_'.join(['', key])
            setattr(getattr(self, '_'.join(['', name])), key, val)

    def _get_slices(self):

        n_c    = [p.order+1 for p in self.trend]
        n_p    = len(self.trend)
        start  = 0
        slices = []

        for i in range(n_p):
            slices.append(slice(start, start + n_c[i]))
            start += n_c[i]

        return slices

class Dataset(object):
    def __init__(self, lc=[], rv=[]):
        """
        class to hold the full dataset to fit.

        params:
            lc : list
               > amelie.data.Photometry objects that hold photometric
                 data.

            rv : list
               > amelie.data.RadialVelocity objects that hold radial velocity
                 data
        """

        self.lc = lc
        self.rv = rv

#        if self.n_lc > 0:
#            self.time_lc  = [p.time for p in self.lc_list]
#            self.flux     = [p.flux for p in self.lc_list]
#            self.flux_err = [p.flux_err for p in self.lc_list]
#            self.texp     = [p.texp for p in self.lc_list]
#
#        # RV data is concatenated and sorted in time so that models that span
#        # multiple datasets are easier to compute, e.g. GPs and RV drifts. 
#        # Individual datasets are referred to by their indices; self.index_rv
#        # that matches the corresponding values in self.time_rv
#        if self.n_rv > 0:
#            # sort times
#            time_rv      = np.concatenate([p.time for p in self.rv_list])
#            t_argsort    = np.argsort(time_rv)
#
#            # argsort the argsort (yo_dawg.jpg) to find back to original indices
#            len_rv = [len(p.time) for p in self.rv_list]
#            len_rv.insert(0,0)
#
#            t_double_argsort  = np.argsort(t_argsort)
#            self._index_rv    = []
#            for i in range(1,len(len_rv)):
#                self._index_rv.append(t_double_argsort[len_rv[i-1]:
#                                                       len_rv[i-1]+len_rv[i]])
#
#            self.time_rv = time_rv[t_argsort]
#            self.rv      = np.concatenate([p.rv for p in self.rv_list])[t_argsort]
#            self.rv_err  = np.concatenate([p.rv_err for p in self.rv_list])[t_argsort]

#    @property
#    def index_rv(self):
#        return self._index_rv
    def get_observed_light_curve(self, name=None):
        if name is None:
            x = np.concatenate([d.x for d in self.lc])
            y = np.concatenate([d.y for d in self.lc])
            yerr = np.concatenate([d.y_err for d in self.lc])
        else:
            d = [d for d in self.lc if d.channel == name][0]
            x = d.x
            y = d.y
            yerr = d.y_err

        return x, y, yerr

    def get_observed_radial_velocity(self, name=None):
        if name is None:
            x = np.concatenate([d.x for d in self.rv])
            y = np.concatenate([d.y for d in self.rv])
            yerr = np.concatenate([d.y_err for d in self.rv])
        else:
            d = [d for d in self.rv if d.channel == name][0]
            x = d.x
            y = d.y
            yerr = d.y_err

        return x, y, yerr
       
    @property
    def n_lc(self):
        return len(self.lc)

    @property
    def n_rv(self):
        return len(self.rv)

    def add_light_curve(self, *args, **kwargs):
        self.lc.append(Data(*args, **kwargs))

    def add_radial_velocity(self, *args, **kwargs):
#        _bad_kwargs = ["in_transit"]
        if "in_transit" in kwargs:
            raise TypeError("add_radial_velocity() got an unexpected keyword "
                            "argument ``in_transit``")
        self.rv.append(Data(*args, **kwargs))

