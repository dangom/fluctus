.. raw:: html

         <h1 align="center">Fluctus</h1>
         <p align="center">
         <img width="256px" src="img/banner.png" alt="Banner">
         </p>


.. contents:: **Table of Contents**
    :backlinks: none


Fluctus provides a single interface for handling oscillatory data.
This interface is the class Oscillation, which offers methods for
normalizing samples, averaging them, interpolating them to a different grid,
discarding initial transient timepoints and computing a frequency spectrum.
It also offers methods for estimating phase and amplitude, and plotting the data.

This package is WIP and the interface and calling conventions are expected to
change as I learn what feels clumsy or confusing.

.. code-block:: python
                
    from fluctus.interfaces import Oscillation
    osc = Oscillation(tr=1.0, period=20.0, stimulus_offset=14.0)
    # Interpolate to a 100ms grid and percent signal change normalize
    osc.interp(target_sampling_out=0.1).psc()
    # See all operations that were done to the data
    print(osc.transformation_chain)
    # Get the transformed data
    osc_preproc = osc.transformed_data

Since my use case is dealing with neuroimaging data, it also provides methods
for reading data from nifti files given a mask.

.. code-block:: python

    osc = Oscillation.from_nifti("mask.nii.gz", "data.nii.gz", period=20.0)

Other utility classes in this package are models for generating HRFs and convolving
responses.

Installation
------------

fluctus is distributed on `PyPI <https://pypi.org>`_ as a universal
wheel and is available on Linux/macOS and Windows and supports
Python 3.7+.

.. code-block:: bash

    $ pip install fluctus

The optinal WIP viz module requires ants, which is not listed as a requirement.

License
-------

fluctus is distributed under the terms of both

- `MIT License <https://choosealicense.com/licenses/mit>`_
- `Apache License, Version 2.0 <https://choosealicense.com/licenses/apache-2.0>`_

at your option.
