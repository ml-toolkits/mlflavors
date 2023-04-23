Examples
========
This section provides a usage example for each flavor.
All modules referenced in the examples below can be found in the
`examples <https://github.com/ml-toolkits/mlflavors/tree/main/examples>`_ folder
of the github repository.

The interface for utilizing the model as a ``pyfunc`` type for generating predictions
uses a *single-row* ``Pandas DataFrame`` configuration argument. Refer to the
`API documentation <https://mlflavors.readthedocs.io/en/latest/index.html>`_
for a description of the supported columns in this configuration ``Pandas DataFrame``.

.. include:: examples/orbit.rst

.. include:: examples/sktime.rst

.. include:: examples/statsforecast.rst

.. include:: examples/pyod.rst

.. include:: examples/sdv.rst
