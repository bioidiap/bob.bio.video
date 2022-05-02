.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Mon 13 Aug 2012 12:36:40 CEST

.. _bob.bio.video:

========================================
 Run Video Face Recognition Experiments
========================================

This package is part of the ``bob.bio`` packages, which provide open source tools to run comparable and reproducible biometric recognition experiments.
In this package, tools to run video face recognition experiments are provided.

For more detailed information about the structure of the ``bob.bio`` packages, please refer to the documentation of :ref:`bob.bio.base <bob.bio.base>`.

In the following, we provide more detailed information about the particularities of this package only.

Get Started (TLTR)
==================

To run biometric experiments using the :doc:`pipeline_simple_intro` with video databases, please make usage of the `video-wrapper` `entry-point <https://packaging.python.org/specifications/entry-points/>`_.
For instance the example below uses the `video-wrapper` to run face recognition experiments using one of our baselines from :ref:`bob.bio.face <bob.bio.face>` and the Youtube Face datase::



$ bob bio pipeline simple youtube iresnet100 video-wrapper



Please, go through the documentation of this package and :ref:`bob.bio.base <bob.bio.base>` to see how these commands work.


Users Guide
===========

.. toctree::
   :maxdepth: 2

   faq
   annotators

Reference Manual
================

.. toctree::
   :maxdepth: 2

   implemented

ToDo-List
=========

This documentation is still under development.
Here is a list of things that needs to be done:

.. todolist::


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. todolist::
