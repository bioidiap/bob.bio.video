.. _bob.bio.video.faq:

================================
Frequently Asked Questions (FAQ)
================================



How to change the way frames are selected in my experiment?
-----------------------------------------------------------

The default frame selector in this package :any:`bob.bio.video.select_frames` allows you to select the
way frames are select `first`, `spread`, `step`, and `all` the the maximum number of frames can be used in this select.
The examples below shows some examples on how to use this selector.


Select the first frame only from every video
............................................

.. code-block:: python

    >>> from bob.bio.video import select_frames
    >>> from functools import partial
    >>> frame_selector = partial(select_frames, selection_style="first", max_number_of_frames=1)
    >>> frame_indices = [] # Some arbitrary list holding the frame indices
    >>> selected_frames = frame_selector(frame_indices)


Select all frames
.................

.. code-block:: python

    >>> from bob.bio.video import select_frames
    >>> from functools import partial
    >>> frame_selector = partial(select_frames, selection_style="all", max_number_of_frames=None)
    >>> frame_indices = [] # Some arbitrary list holding the frame indices
    >>> selected_frames = frame_selector(frame_indices)


Select all frames, but with an upper-bound of 100 frames
........................................................

.. code-block:: python

    >>> from bob.bio.video import select_frames
    >>> from functools import partial
    >>> frame_selector = partial(select_frames, selection_style="all", max_number_of_frames=100)
    >>> frame_indices = [] # Some arbitrary list holding the frame indices
    >>> selected_frames = frame_selector(frame_indices)


Select 10 frames equally spread from the whole video
....................................................

.. code-block:: python

    >>> from bob.bio.video import select_frames
    >>> from functools import partial
    >>> frame_selector = partial(select_frames, selection_style="spread", max_number_of_frames=10)
    >>> frame_indices = [] # Some arbitrary list holding the frame indices
    >>> selected_frames = frame_selector(frame_indices)


Now that I have customized my frame selector, so what?
......................................................


Once this frame selector is set, you can customize your experiment to use it.
The example below shows how to customize it for the YouTube Video Faces dataset (using one of the examples above)


.. code-block:: python

    >>> from bob.bio.video.database import YoutubeDatabase
    >>> from functools import partial
    >>> from bob.bio.video.utils import select_frames

    >>> frame_selector = partial(select_frames, selection_style="spread", max_number_of_frames=10)
    >>> database = YoutubeDatabase(protocol="fold0", frame_selector=frame_selector)


Once this is saved into a python file (e.g. `my-dataset.py`), the PipelineSimple (:doc:`pipeline_simple_intro`) can be triggered as::


 $ bob bio pipeline simple my-dataset.py [BASELINE] video-wrapper
