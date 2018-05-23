
======================
Implementation Details
======================

Wrapper classes
---------------

The tools implemented in this package provide wrapper classes for preprocessing, feature extraction and face recognition algorithms that are implemented in other packages of ``bob.bio``.
The basic idea is that the wrapped algorithms are provided with several frames of the video.
For this purpose, the :py:class:`bob.bio.video.FrameSelector` can be applied to select one or several frames from the source video.
For each of the selected frames, the faces are aligned -- either using hand-labeled data, or after detecting the faces using :py:class:`bob.bio.face.preprocessor.FaceDetect`.
Afterward, features are extracted, models are enrolled using several frames per video, and the scoring procedure fuses the scores from one model and several probe frames of a probe video.
If one of the base algorithms requires training, the wrapper classes provide these information accordingly.

Hence, in this package we provide three wrapper classes:

* :py:class:`bob.bio.video.preprocessor.Wrapper`
* :py:class:`bob.bio.video.extractor.Wrapper`
* :py:class:`bob.bio.video.algorithm.Wrapper`

Each of these wrapper classes is created with a base algorithm that will do the actual preprocessing, extraction, projection, enrollment or scoring.
The base class can be specified in three different ways.
The most prominent ways will surely be to use some of the registered :ref:`bob.bio.base.resources`.
The more sophisticated way is to provide an *instance* of the wrapped class, or even a *string* that represents a constructor call of the desired object.
Finally (rarely used, though) you can provide the path of :ref:`bob.bio.base.configuration-files`.

The IO of the preprocessed frames, and extracted or projected features is provided using the :py:class:`bob.bio.video.FrameContainer` interface.
This frame container reads and writes :py:class:`bob.io.base.HDF5File`\s, where it stores information about the frames.
Additionally, it uses the IO functionality of the wrapped classes to actually write the data in the desired format.
Hence, all IO functionalities of the wrapped classes need to be able to handle :py:class:`bob.io.base.HDF5File`.

.. note::
   The video extensions also integrate into the specialized scripts provided by :ref:`bob.bio.gmm <bob.bio.gmm>`.


.. _bob.bio.video.resources:

Registered Resources
--------------------


In this package we do not provide registered resources for the wrapper classes.
Hence, when you want to run an experiment using the video wrapper classes, you might want to create the wrapper classes inline:

.. code-block:: sh

   verify.py --database youtube --preprocessor 'bob.bio.video.preprocessor.Wrapper("landmark-detect")' --features 'bob.bio.video.extractor.Wrapper("dct-blocks")' --algorithm 'bob.bio.video.algorithm.Wrapper("gmm")' ...


.. _bob.bio.video.databases:

Databases
~~~~~~~~~

All video databases defined here rely on the :py:class:`bob.bio.base.database.BioDatabase` interface, which in turn uses the `verification_databases <https://www.idiap.ch/software/bob/packages>`_.

After downloading and extracting the original data of the data sets, it is necessary that the scripts know, where the data was installed.
For this purpose, the ``verify.py`` script can read a special file, where those directories are stored, see :ref:`bob.bio.base.installation`.
By default, this file is located in your home directory, but you can specify another file on command line.

The other option is to change the directories directly inside the configuration files.
Here is the list of files and replacement strings for all databases that are registered as resource, in alphabetical order:

* MOBIO: ``'mobio-video'``

  - Videos: ``[YOUR_MOBIO_VIDEO_DIRECTORY]``

* Youtube: ``'youtube'``

  - Frames : ``[YOUR_YOUTUBE_DIRECTORY]``

    .. note::
       You can choose any of the frame databases, i.e., the ``frames_images_DB`` directory containing the original data, or the ``aligned_images_DB`` containing pre-cropped faces.


You can use the ``databases.py`` script to list, which data directories are correctly set up.


.. _bob.bio.video.baselines:

Executing Face Baselines using Video databases
----------------------------------------------

There may exist many face recognition baselines (:ref:`bob.bio.face.baselines`)
designed for still images that you may want to try on video databases. An easy
way to do that is to use our ``video-wrapper`` (It's a ``bob.bio.config``
entrypoint). What this does is to take the preprocessor, extractor, and
algorithm from the baseline and wrap it in bob.bio.video classes:

.. code-block:: sh

    $ bob bio baseline <face-baseline> <video-database> video-wrapper
