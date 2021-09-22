#! /usr/bin/env python

from __ future__ import absolute_import

import os

__all__ = ['Files']

class Files(object):
    def __init__(self, parameter_file=None):#run_dir='', run_name=''):
        """
        Class for dealing with the various output files in amelie.

        params:
            parameter_file  : str
                file path to parameter file in YAML format, must have .yaml
                extension

            run_dir    : str (default '')
                path to directory where various runs will be created, e.g.
                ``/path/to/my_runs``

            run_name   : str (default '')
                subdirectory of run_dir where the parameter file can be 
        """

        # get file paths
        comps         = parameter_file.split(os.sep)
        self.root     = os.path.join(os.sep, *comps[:-1])
        par_file     = comps[-1]

        assert par_file.endswith('.yaml'), ("input file: {} is not YAML".format(par_file))

        self.parin_file     = self._create_filepath("parameters", "in")
        self.notes_file     = self._create_filepath("notes", "txt")
        self.chain_file     = self._create_filepath("chain", "csv")
        self.walker_file    = self._create_filepath("walkers", "png")
        self.corner_file    = self._create_filepath("corner", "png")
        self.rvfit_file     = self._create_filepath("rvfit", "pdf")
        self.result_file    = self._create_filepath("results", "txt")


    def _create_filepath(self, name, extension):
        """ creates file paths given a name and its extension """
        return os.path.join(self.root, '.'.join([name, ext]))
        

