from __future__ import division
import logging
import sys
import functools

import numpy as np
from scipy.ndimage.interpolation import zoom
from six import iteritems

try:
    sys.path.append('/shared/bioapps/itk/itk_shared/jp2/build')
    import jpeg_twok
except (ImportError, ModuleNotFoundError):
    import glymur

from allensdk.mouse_connectivity.grid import image_utilities as iu
from .base_subimage import PolygonSubImage, SegmentationSubImage, IntensitySubImage


#==============================================================================


class CavSubImage(PolygonSubImage):

    required_polys = ['missing_tile', 'cav_tracer']


    def compute_coarse_planes(self):

        nonmissing = np.logical_not(self.images['missing_tile'])
        del self.images['missing_tile']

        self.apply_pixel_counter('sum_pixels', nonmissing)

        cav_nonmissing = np.multiply(self.images['cav_tracer'], nonmissing)
        del nonmissing
        self.apply_pixel_counter('cav_tracer', cav_nonmissing)
        del cav_nonmissing

        del self.images


#==============================================================================
