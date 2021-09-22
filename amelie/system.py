#! /usr/bin/env python

from __future__ import print_function, division, absolute_import

import numpy as np
import ellc
import emcee
import datetime
from collections import OrderedDict
from scipy.optimize import minimize, curve_fit, differential_evolution
from scipy.interpolate import BSpline, splrep
from data_tess2 import Dataset
from multiprocessing import Pool
import tqdm
#from model import Polynomial
import sys
import os
import matplotlib.pyplot as plt


# CHANGELOG
# NOTE: VEDAD!!!!! ellc.rv now returns both primary and secondary RV

def _log_probability_picklable(theta, instance, name):
    """
    Method for multiprocessing to work with class instance methods (making it
    pickleable).
    """
    return getattr(instance, name)(theta)

class System(object):
    def __init__(self, primary, companions=[]):

        self.primary    = primary
        self.companions = companions
        self.companion = {c.name : c for c in self.companions}
        self.data = Dataset()
        
        self.free_parameters = self._has_error()
        self.ndim = len(self.free_parameters)
        self._initial_values = self._get_all_values()

        self.verbose = False




    def get_channel(self, *args):
        return self.primary.get_channel(*args)
    
    def _has_error(self):
        free_parameters = []

        for companion in self.companions:
            name = companion.name

            # loop through the free_parameters that are allowed to be set
            for p in companion.allowed_set_parameters:

                # check if that parameter has a specified error, and if so add
                # it to the list of free parameters
                if getattr(getattr(companion, p), 'error') is not None:
                    free_parameters.append(':'.join([companion.fname, p]))
        
        # limb darkening coefficients
        for channel, ld in self.primary.channels.items():
            for p in self.primary._allowed_priors:
                if getattr(getattr(ld, p), 'error') is not None:
                    free_parameters.append(':'.join([ld.fname, p]))

        # jitter terms
        for d in (self.data.lc + self.data.rv):
#        for d in self.data.lc:
            print('checking data')
            for p in d._allowed_priors:
                print('jitter obj', getattr(getattr(d, p), "error"))
                if getattr(getattr(d, p), "error") is not None:
                    free_parameters.append(":".join(["channel", d.channel, "jitter"]))


        return tuple(free_parameters)
##        print(self.free_parameters)
#        self.ndim            = len(self.free_parameters)


    def _has_prior(self):
        """ get dict of companions and their parameters that have set priors, 
            to be iterated in `log_probability`.
        """
        prior_parameters = OrderedDict()

        # just... don't ask

        # orbital parameters
        for c in self.companions:
            prior_parameters[c.fname] = []

            # check that photometry is being fit
            if (c.rsum.value is not None) and (c.k is not None):

                # if so, loop through the allowed photometric priors
                for p in c.allowed_lc_priors:

                    # check if bounds is given
                    if getattr(getattr(c, p), 'bounds') is not None:
                        prior_parameters[c.fname].append(p)

                    # check if the normal prior is given
                    elif ((getattr(getattr(c, p), 'mu') is not None) and
                          (getattr(getattr(c, p), 'sd') is not None)):
                        prior_parameters[c.fname].append(p)

            # check if RV is being fit
            if c.Kpri.value is not None:

                for p in c.allowed_rv_priors:

                    if getattr(getattr(c, p), 'bounds') is not None:
                        prior_parameters[c.fname].append(p)

                    elif ((getattr(getattr(c, p), 'mu') is not None) and
                          (getattr(getattr(c, p), 'sd') is not None)):
                        prior_parameters[c.fname].append(p)

            # remove duplicate parameters if both RV and photometry is fit
#            prior_parameters[c.fname] = tuple(set(prior_parameters[c.fname]))
            prior_parameters[c.fname] = list(np.unique(prior_parameters[c.fname]))

        # stellar parameters
        # check if any channels have been specified and loop through them
        if any(self.primary.channels):
#            print('should enter')
#            print(self.primary.channels, type(self.primary.channels))
#            print(self.primary.channels.items())
#            print('prior params', prior_parameters)
            for c,ld in self.primary.channels.items():
#                print(c, ld.fname)
                prior_parameters[ld.fname] = []
#                prior_parameters[':'.join(['channel', c])] = []

                for p in self.primary._allowed_priors:

                    # check if current parameter has a set prior
                    if getattr(getattr(self.primary.get_channel(c), p), 'bounds') is not None:
                        prior_parameters[ld.fname].append(p)
#                        prior_parameters[":".join(["channel", c])].append(p)

                    elif ((getattr(getattr(self.primary.get_channel(c), p),
                                  'mu') is not None) and 
                          (getattr(getattr(self.primary.get_channel(c), p),
                                   'sd') is not None)):
                        prior_parameters[ld.fname].append(p)
#                        prior_parameters[":".join(["channel", c])].append(p)

#                prior_parameters[ld.fname] = tuple(prior_parameters[ld.fname])
#                prior_parameters[":".join(["channel", c])] = (
#                        tuple(prior_parameters[":".join(["channel", c])]))
#                print(prior_parameters)
                # wtf is this line for???
                prior_parameters[":".join(["channel", c])] = (
                        prior_parameters[":".join(["channel", c])])

#                prior_parameters[":".join(["channel", c])] = (
#                        prior_parameters[":".join(["channel", c])])

        for d in (self.data.lc + self.data.rv):

            fname = ":".join(["channel", d.channel]) 
            if fname not in prior_parameters:
                prior_parameters[fname] = []

            for p in d._allowed_priors:

                if getattr(getattr(d, p), 'bounds') is not None:
                    prior_parameters[fname].append(p)

                elif ((getattr(getattr(d, p), 'mu') is not None)
                        and 
                      (getattr(getattr(d, p), 'sd') is not None)):
                    prior_parameters[fname].append(p)
#            prior_parameters[fname] = tuple(prior_parameters[fname])
            prior_parameters[fname] = prior_parameters[fname]

#        print('prior params', prior_parameters)
        return prior_parameters
    

#    def _get_n_int(self, t, P, Tpri, W, oversample=1, in_transit=True, f=2):
#        """
#        get n_int array for ellc.lc() input.
#        """
#        f /= 100 + 1
#        m = np.zeros_like(t, dtype=int) + int(oversample)
#        if in_transit:
#            phase = (t - Tpri) % P / P
#            phase[phase > 0.5] -= 1
#            out = ~np.logical_and(t > -f*0.5*W, t < f*0.5*W)
#            m[out] = 0
#        return m

#    def get_model(self, func, t, companion,
#                  ld=None, ldc=None,
#                  oversample=1, t_exp=None, in_transit=False, 
#                  func_kwargs=None):
#        """ get the specified model (ellc.lc() or ellc.rv())
#            for the specified companion. Common function because most parameters
#            overlap between the two functions """
#
#        c = companion
#        n_int = self._get_n_int(t, c.P.value, c.Tpri.value, c.W.value,
#                                oversample=oversample, in_transit=in_transit)
#
#        ellc_kwargs  = {'radius_1' : c.r_1.value,
#                        'radius_2' : c.r_2.value,
#                        'incl'     : c.incl.value,
#                        'sbratio'  : c.J.value,
#                        'period'   : c.P.value,
#                        't_zero'   : c.Tpri.value,
#                        'q'        : c.q.value,
#                        'ld_1'     : ld,
#                        'ldc_1'    : ldc,
#                        't_exp'    : t_exp, 
#                        'n_int'    : n_int,
#                        'grid_1'   : 'very_sparse',
#                        'grid_2'   : 'very_sparse'}
#
#        ellc_kwargs.update(func_kwargs)
#
#        return func(t, **ellc_kwargs)

#    def get_light_curve(self, t, companions, **kwargs):
#
##        if isinstance(companion, Companion):
##            companion = [companion]
#
#        mod = np.ones_like(t)
#        for c in companions:
#            func_kwargs = {'func_kwargs': {'light_3' : c.light_3.value}}
##            all_kwargs  = {'func_kwargs': {'light_3' : c.light_3.value}}
##            all_kwargs.update(kwargs)
#            kwargs.update(func_kwargs)
#            
#            mod *= self.get_model(ellc.lc, t, c,# func_kwargs=func_kwargs,
##                                  **all_kwargs)
#                                  **kwargs)
#
#        return mod
#        return self.get_model(ellc.lc, *args, **all_kwargs)


    def get_radial_velocity(self, t, companions):
        # ellc.rv() parameters

        # NOTE: VEDAD!!!!! ellc.rv now returns both primary and secondary RV
#        mod = np.zeros_like(t)
        mod = np.zeros((2, len(t)))
        for c in companions:

            ellc_kwargs  = {'radius_1' : c.r_1.value,
                            'radius_2' : c.r_2.value,
                            'incl'     : c.incl.value,
                            'sbratio'  : c.J.value,
                            'period'   : c.P.value,
                            't_zero'   : c.Tpri.value,
                            'q'        : c.q.value,
                            'f_s'      : c.f_s.value,
                            'f_c'      : c.f_c.value,
                            'grid_1'   : 'very_sparse',
                            'grid_2'   : 'very_sparse',
                            'a'        : c.a.value,
                            'q'        : c.q.value,
                            'flux_weighted' : False}


#            func_kwargs = {'func_kwargs' : {}}
#            func_kwargs  = {'func_kwargs': {'a'             : c.a.value,
#                                            'q'             : c.q.value,
#                                            'vsini_1'       : c.vsini_1.value,
#                                            'lambda_1'      : c.lambda_1.value,
#                                            'flux_weighted' : flux_weighted}}
            grid = c._warp_times(t)

            # NOTE: VEDAD!!!!! ellc.rv now returns both primary and secondary RV
            mod += np.array(ellc.rv(grid, **ellc_kwargs))#[0]
#            kwargs.update(func_kwargs)
#            mod += self.get_model(ellc.rv, t, c, **kwargs)[0]
        
#        all_kwargs.update(kwargs)
        return mod
#        return self.get_model(ellc.rv, *args, **all_kwargs)

    @property
    def log_prior(self):
        """ get the total log prior probability of the system """

        l = 0


        # loop through companions that have specified priors
#        print('has_prior', self._has_prior_dict)
        for c,p in self._has_prior_dict.items():

#            if c.startswith('companion'):
#                # select the current companion
#                cc = [x for x in self.companions if x.fname == c][0] # select actual object

                # loop through its prior variables and sum up the log_prior
#                for pp in p:
#                    if (self.verbose and np.isinf(getattr(getattr(cc, pp), 'log_prior'))):
#                        print("{:s} with value {:.7f} is out of bounds {:}".format(cc.fname, 
#                                            getattr(getattr(cc, pp), 'value'),
#                                            getattr(getattr(cc, pp), 'bounds')))
#
#                    l += getattr(getattr(cc, pp), 'log_prior')

#            elif c.startswith('channel'):
#
#                # select the current limb darkening object
#                cc = [x for x in self.primary.channels.values() 
#                      if  x.fname == c][0] # select actual LimbDarkening object

#                for pp in p:
#                    if (self.verbose and np.isinf(getattr(getattr(cc, pp), 'log_prior'))):
#                        print("{:s} with value {:.7f} is out of bounds
#                                {:}".format(cc.fname, 
#                                            getattr(getattr(cc, pp), 'value'),
#                                            getattr(getattr(cc, pp), 'bounds')))
#
#                    l += getattr(getattr(cc, pp), 'log_prior')

            # get companion object that parameter belongs to
#            cc = self._match_name(c)
#            print('c', c)

            # loop through specific companion parameters with prior
            for pp in p:
                cc = self._match_name(":".join([c, pp]))
                if (self.verbose and np.isinf(getattr(getattr(cc, pp), 'log_prior'))):
                    print("{:s} with value {:.7f} is out of bounds {:}".format(
#                        ":".join([cc.fname, pp]),
                        ":".join([c, pp]),
                        getattr(getattr(cc, pp), 'value'),
                        getattr(getattr(cc, pp), 'bounds')))

                # a convoluted way to get Companion().Parameter().log_prior
                l += getattr(getattr(cc, pp), 'log_prior')

        return l

    def get_parameter(self, companion, parameter, attr=None):
        if attr is None:
            return getattr(companion, parameter)
        else:
            return getattr(getattr(companion, parameter), attr)

    def chi_square(self, data, model, error):
        return np.sum(((data - model) / error)**2)

    def log_likelihood(self, data, model, error):
        # check what the final term of the likelihood should be
        inv_sigma2 = 1/error**2
        return -0.5 * np.sum((data - model)**2 * inv_sigma2 - np.log(inv_sigma2))


    def _set_all_parameters(self, theta):

        for pc,v in zip(self.free_parameters, theta):

            # get companion names and parameter names
            # e.g. companion:Earth:k -> companion:Earth and k
#            s = pc.split(':')
#            cname, pname = ':'.join(s[:-1]), s[-1]

            # find matching object
#            c = self._match_name(cname)
            c = self._match_name(pc)

            pname = pc.split(":")[-1]
            c.set_parameter(pname, value=v)

    def set_parameter_dict(self, theta):
        for p, v in theta.items():
            s = p.split(":")
            cname, pname = ":".join(s[:-1]), s[-1]
            c = self._match_name(cname)
            c.set_parameter(pname, value=v)

    def optimize(self, start=None, vars=None, method='de', workers=1,
                verbose=False):

        self.free_parameters = self._has_error()
        self.ndim = len(self.free_parameters)
        self._initial_values = self._get_all_values()

        # fit for all free parameters if none are specified
        if vars is None:
            vars = [x.split(":")[-1] for x in self.free_parameters]

        print("optimizing logp for variables: [{:s}]".format(", ".join(vars)),
                file=sys.stderr)

        # provide starting values if none is specified
        if start is None:
            # _has_prior() needs to be called before _get_test_point()
            self._has_prior_dict = self._has_prior()
            start = OrderedDict(zip(self.free_parameters,
                                    self._get_test_point()))

        print('start', start)
        vars_index = []
        bounds     = []

        # find the indices of parameters that are being fit for
        for p in vars:
            for c in self.companions:
                fname = ":".join([c.fname, p])
                vars_index.append(self.free_parameters.index(fname))
                bounds.append(self.get_parameter(c, p, attr='bounds'))
        vars_index = np.array(vars_index)
        
        # initial parameter vector
        theta      = self._dict_to_array(start)
        x0         = theta[vars_index]
        logp_start = self.log_probability(theta, check_prior=False)

#        print('logp start', logp_start)
        assert np.isfinite(logp_start), ("initial logp is nan, check input values"
                                      " to models")

#        print('bounds', bounds)
        if verbose:
            bar = tqdm.tqdm()

        # minimization function
        def _neg_lnprob(sub_vars, all_vars):
            all_vars[vars_index] = sub_vars

            res = -self.log_probability(all_vars, check_prior=False)

            if verbose:
                bar.set_postfix(logp="{0:e}".format(-res))
                bar.update()
            return res

        # elapsed time
        tstart = datetime.datetime.today()

        # do the fit
#        res   = minimize(_neg_lnprob, x0=x0, args=(theta,), method="L-BFGS-B",
#                bounds=bounds, options={'disp':True, 'eps':1e-8})
        res   = differential_evolution(_neg_lnprob, args=(theta,),
                        bounds=bounds, workers=workers)

        if verbose:
            bar.close()

        print(res)
        tend   = datetime.datetime.today()

        # get status
        success = res.success
        niters  = res.nit

        # update solution if successful
        map_soln   = start

        if success:
            # loop through variables to fit...
            for varname, varval in zip(vars, res.x):

                # for each companion... 
                for c in self.companions:

                    # replace the dict entry with the new fitted value 
                    fname = ":".join([c.fname, varname])
                    map_soln[fname] = varval

            # update all parameters and check new logp
            logp_end = self.log_probability(self._dict_to_array(map_soln),
                                            check_prior=False)

        # if not successful, stick to the initial solution
        else:
            logp_end = self.log_probability(self._dict_to_array(map_soln),
                                            check_prior=False)

        # format output to screen
        def _time_format(td):
            minutes, seconds = divmod(td.seconds + td.days * 3600, 60)
            return "{:02d}:{:02d}".format(minutes, seconds)

        _msg = ""

        if success:
#            _msg += "message: optimization successful\n"
            _msg += "message: {0}\n".format(res["message"])
        else:
            _msg += ("message: Optimization failed, something went wrong. " 
                     "Returning to initial values.\n"
                     "`minimize` output: {0}\n".format(res['message']))

        _msg += ("[nit: {:d}   in   {:s}]\n"  
                 "logp: {:.10f} -> {:.10f}".format(niters,
                                                   _time_format(tend-tstart),
                                                   logp_start, -res["fun"]
                                                   )
                 )
        print(_msg, file=sys.stderr)

        return map_soln


    @staticmethod
    def _poly_func(*c, d=None, size=None):

#        xvec    = np.atleast_2d(xvec)
        polyval = np.zeros(size)

        for p, sl in zip(d.trend, d.slices):
            p._set_all_parameters(c[sl])
            polyval += p.eval()

        return polyval



    def set_all_parameters(self, theta):
        self.log_probability(theta)
            
    def log_probability(self, theta, check_prior=True):


        # set the new values of theta to the right companion and parameter
        self._set_all_parameters(theta)

        if check_prior:
            logp = self.log_prior

            # check if parameter out of bounds
            if np.isinf(logp):
                return logp
        else:
            logp = 0 

        # next, calculate model
#        for c in self.companions:
#        if self._fit_light_curve:
        for d in self.data.lc:
            mod = (self.get_channel(d.channel).
                    get_light_curve(
                        d.x,
                        self.companions,
                        oversample = d.oversample,
                        t_exp      = d.t_exp,
                        in_transit = d.in_transit
                        )
                    )

#            size = int(0.5 / d.t_exp)
#            if not size % 2: # needs to be odd
#                size += 1
            y_err = np.sqrt(d.y_err**2 +  (mod * d.jitter.value)**2)

#            trend = self.splinef(d.y/mod)
            if d.knot_spacing is not None:
                m = np.any([~c._in_transit_mask(d.x) for c in self.companions],
                        axis=0)
                trend = self.get_spline(d.x,
                        d.y/mod,
#                        d.y_err, 
                        y_err, 
                        mask=m,
                        knot_spacing=d.knot_spacing)(d.x)
                mod *= trend


            else:
                mod *= (np.zeros(len(d.x)) +
                        np.average(d.y / mod, weights=1/y_err**2))
#                            weights=1/(d.y_err**2 + d.jitter.value**2)))

            # calculate log likelihood
            y_err = np.sqrt(d.y_err**2 +  (mod * d.jitter.value)**2)


            logp += self.log_likelihood(d.y, mod, y_err,
#                                        np.sqrt(d.y_err**2 +
#                                                d.jitter.value**2)
                                        )

            if not np.isfinite(logp):
                print('nan in light curve for {:}: '.format(d.channel),
                        self.free_parameters, theta)
                return np.nan


                # additional baseline stuff (e.g. GP)
#                    d.resid = d.y / mod
#                    logp += d.gp(
#        if self._fit_radial_velocity:
        for d in self.data.rv:
            mod1, mod2 = self.get_radial_velocity(d.x, self.companions)

            # Rossiter-McLaughlin model
            if d.flux_weighted:
                keplerian = mod1.copy()

                mod1 += (self.get_channel(d.channel).
                        get_rossiter_mclaughlin(d.x, self.companions,
                                                oversample    = d.oversample,
                                                t_exp         = d.t_exp,
                                                keplerian     = keplerian
                                                )[0]
                        )

            # offsets and polynomial baselines
            if d.trend is not None:
#                xvec = np.vstack([p.x for p in d.trend])
##                f    = lambda x, *c, d=d: System._poly_func(*args, **kwargs)
                if any([p.order > 0 for p in d.trend]):
                    f    = lambda x, *c: System._poly_func(*c, d=d, size=len(d.x))
                    p0   = tuple([p.cvals for p in d.trend])
                    popt, _ = curve_fit(f, d.x, d.y - mod1, sigma=d.y_err,
                                        p0=p0, method='lm')
                    mod1 += f(d.x, *popt)
                else:
#                    mean = np.average(d.y - mod, weights=1/(d.y_err**2))
#                    [p._set_all_parameters([mean]) for p in d.trend]
#                    mod += np.zeros_like(d.x) + mean
                    mod1 += (np.zeros(len(d.x)) +
                            np.average(d.y - mod1, weights=1/(d.y_err**2)))
            else:
                    mod1 += (np.zeros(len(d.x)) +
                            np.average(d.y - mod1, weights=1/(d.y_err**2)))

#                    mean = np.average(d.y - mod, weights=1/(d.y_err**2))
#                    [p._set_all_parameters([mean]) for p in d.trend]
#                    mod += np.zeros_like(d.x) + mean


            logp += self.log_likelihood(d.y, mod1,
                                        np.sqrt(d.y_err**2 +
                                                d.jitter.value**2)
                                        )
#                    d.resid = d.y - mod
            if not np.isfinite(logp):
                print('nan in RV model for {:}'.format(d.channel),
                        self.free_parameters, theta)
                return np.nan

#        print('logp', logp) 
        return logp 

    def _get_all_attr(self, attr):
        """ get all free parameters attributes """
        p0 = []
        for pc in self.free_parameters:

            #remove parameter name at the end
#            x = ':'.join(pc.split(':')[:-1])
#            x = pc

            # get object matching string name 
#            c = self._match_name(x)
            c = self._match_name(pc)

            # get parameter name without the prefixes
#            p = pc.replace(c.fname + ':', '')
#            p = pc.strip(":")[-1]
            p = pc.split(":")[-1]

            # add current attribute to list
#            print(c, p)
            p0.append(getattr(getattr(c, p), attr))

        return OrderedDict(zip(self.free_parameters, p0))

    def _match_name(self, name):
        """ get object matching string name """
        # not proud of this one...

        if name.startswith('companion'):
            fname = ":".join(name.split(":")[:-1])
#            print('name', name)
#            print('fname', fname)
            return [x for x in self.companions if x.fname == fname][0]

#        elif name.startswith("channel") and name.endswith("jitter"):
        elif name.endswith("jitter"):
            channel = name.split(":")[1]
#            print('cname', channel)
            lc = [x for x in self.data.lc if x.channel == channel]
#            print('lc', lc)
            if len(lc) > 0:
                return lc[0]
            else:
#                print('rv', [x for x in self.data.rv if x.channel == channel][0])
                return [x for x in self.data.rv if x.channel == channel][0]

        elif name.startswith('channel'):
            fname = ":".join(name.split(":")[:-1])
            return [x for x in self.primary.channels.values() if x.fname == fname][0]

    def _get_all_values(self):
        """ get all free parameters values """
        return self._get_all_attr('value')

    def _get_all_errors(self):
        """ get all free parameters errors """
        return self._get_all_attr('error')

    def _dict_to_array(self, d):
        """ get values from OrderedDict or dict """
        return np.array(list(x[1] for x in d.items()))

    def _get_test_point(self, start=None):
        """ get a random parameter sample vector given starting values and
        errors """

        if start is None:
            values = self._dict_to_array(self._initial_values)
        else:
            values = self._dict_to_array(start)
        errors = self._dict_to_array(self._get_all_errors())

#        print(len(values), len(errors))
#        print(values, errors)
        pos = values + errors * np.random.randn(self.ndim)

        # set the parameter vector to calculate the prior
        self._set_all_parameters(pos)

        # loop until all parameters are within bounds
        while np.isinf(self.log_prior):
            pos = values + errors * np.random.randn(self.ndim)
            self._set_all_parameters(pos)

        return pos

         
    @property
    def _fit_light_curve(self):
        return self.data.lc > 0

    @property
    def _fit_radial_velocity(self):
        return self.data.rv > 0

    def get_spline(self, x, y, y_err, mask=None, knot_spacing=0.5,
                   debug=False):

        # mask the primary and secondary eclipses
        if mask is None:
            m = np.ones_like(x, dtype=bool)
        else:
            m = mask

        nknots = int(np.ceil((x[-1] - x[0]) / knot_spacing))
        lenx   = len(x)

        # create knot positions at timestamps spaced `knot_spacing` apart
        # and make sure indices [1, -2] have knots
        knot_positions = np.concatenate([
                np.arange(1, lenx-1, int((lenx-2) / nknots)), [lenx-2]])
        knots = x[knot_positions]

        # make sure the two points on either end are unmasked, otherwise spline
        # will throw an error
        m[:2]  = True
        m[-2:] = True

        # finally we make sure that we only keep knots that contain unmasked
        # data (for example if the eclipse width exceeds the knot spacing)
        keep_these = np.array([i for i in range(len(knots)-1)
                if np.any(
                    np.logical_and(x[m] > knots[i], x[m] < knots[i+1]
                        )
                    )
                ]
                )


        knots = knots[keep_these] 

#        m[knot_positions[0]]    = True
#        m[knot_positions[0]-1]  = True
#        m[knot_positions[-1]]   = True
#        m[knot_positions[-1]+1] = True

#        idx_knots = np.arange(1, len(x[m])-1, 
        
#        knots = np.arange(x[m][1], x[m][-2], knot_spacing)

#        keep_these = []
#        for i in range(len(knots)-1):
#            if np.any(any(np.logical_and(x[m] > knots[i], x[m] < knots[i+1]))):
#                keep_these.append(i)

        # 1) need one knot at all "edges"-1 where the distance to next timestamp is
        # >knot_spacing/2
        # 2) make sure that if edge knot is in masked timestamp, unmask the
        # timestamp

#        for kno
#            center = x - knot
            # find index to the left and right of knot position
#            left = len(center[center < 0]) - 1
#            right = left + 1
            # check that knot is not inside masked values
#            cont = True
#            if np.min(np.abs(x - knot)) < knot_spacing/2:
#                keep_these.append(i)
#            if all(m[left:right+1]) and (np.min(np.abs(x - knot)) < knot_spacing):
#                keep_these.append(i)
#            if np.min(np.abs(x - knot)) < knot_spacing:
#                keep_these.append(i)
#

#        print(knots)
#        print('mask', m)
#        dx = np.diff(x[m])

        # index where knot 
#        gaps, = np.where(dx > knot_spacing/2)
####        print('gaps', gaps)
###        #print(gaps)
#        knots = []
#        if any(gaps):
#            for gap in gaps:
#                nknots = int(np.ceil




#                knots.append(np.arange(x[m][1], x[m][gap-1], knot_spacing))
#            knots.append(np.arange(x[m][gap+1], x[m][-2], knot_spacing))
###            knots.append(np.arange(x[m][gap+1], x[m][-2], knot_spacing))
##            knots = np.concatenate(knots) # can also just create the linear array and remove points
##        else:
##            knots = np.arange(x[m][1], x[m][-2], knot_spacing)
###
####        print('knots', knots)
#        try:
#            tck = splrep(x[m], y[m], w=1/y_err[m], k=3, task=-1, t=knots)
        tck = splrep(x[m], y[m], w=1/y_err[m], k=3, task=-1, t=knots)
###
#        except ValueError:
##            
#            plt.figure()
#            plt.plot(x, y, '.')
#            plt.plot(knots, np.ones_like(knots), 'o')
#            plt.plot(x[~m], y[~m], 'r.')
#            plt.show()
#            sys.exit()
##
        return BSpline(*tck)


    def update(self):
        self.free_parameters = self._has_error()
        self.ndim = len(self.free_parameters)
#        print(self.free_parameters)
        self._initial_values = self._get_all_values()

        self._has_prior_dict = self._has_prior()

    def sample(self, steps, walkers=None, start=None, threads=1, thin=1,# data=None,
#            fit_light_curve=True, fit_radial_velocity=True, 
            verbose=False):

#        knot_spacing = 0.5
#        time = self.data.lc[0].x
#        flux = self.data.lc[0].y
#        flux_err = self.data.lc[0].y_err
#        time_difference = np.diff(time)
#        gaps = np.where(time_difference > knot_spacing)
#        #print(gaps)
#        knots = []
#        for gap in gaps:
#            knots.append(np.arange(time[1], time[gap-1], knot_spacing))
#        knots.append(np.arange(time[gap+1], time[-2], knot_spacing))
#        knots = np.concatenate(knots) # can also just create the linear array and remove points
#        tck = splrep(time, flux, w=1/flux_err,
#                                    k=3,
#                                    task=-1,
#                                    t=knots)
#        m = np.ones_like(time, dtype=bool)
#        m[0] = False
#        m[-1] = False
#        self.data.lc[0].x = time[m]
#        self.data.lc[0].y = flux[m]
#        self.data.lc[0].y_err = flux_err[m]

#        self.splinef = BSpline(*tck)#, extrapolate=False)



        self.update()
#        self.free_parameters = self._has_error()
#        self.ndim = len(self.free_parameters)
##        print(self.free_parameters)
#        self._initial_values = self._get_all_values()

        if walkers is None:
            walkers = 10 * self.ndim

        if verbose:
            self.verbose = verbose


#        self._has_prior_dict = self._has_prior()
#        self.data       = data

        print("generating initial walker positions...", end="\r",
                file=sys.stderr)
        self.p0 = [self._get_test_point(start=start) for _ in range(walkers)]
        print("generating initial walker positions... done", file=sys.stderr)

#        print("started MCMC at {}".format(start))
#        start = datetime.today()

        plst = []
        for c in self.companions:
            for p in self._get_all_attr('value').keys():
                plst.append(p)
        print("sampling {:d} parameters: [{:s}]".format(self.ndim,
                                                        ", ".join(plst)
                                                        ),
                                                        file=sys.stderr
                                                    )



        if threads > 1:
            os.environ["OMP_NUM_THREADS"] = "1"

            with Pool(processes=threads) as pool:
                sampler = emcee.EnsembleSampler(walkers, self.ndim,
                                                _log_probability_picklable,
                                                args=(self, 'log_probability'),
                                                pool=pool)
                sampler.run_mcmc(self.p0, steps, thin_by=thin, progress=True)
        else:
            sampler = emcee.EnsembleSampler(walkers, self.ndim,
                                            self.log_probability)
            sampler.run_mcmc(self.p0, steps, thin_by=thin, progress=True)


        self.sampler = sampler

        pass
            
    

