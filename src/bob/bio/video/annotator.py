import collections
import logging

import bob.bio.base
import bob.bio.face

from . import utils

logger = logging.getLogger(__name__)


def normalize_annotations(annotations, validator, max_age=-1):
    """Normalizes the annotations of one video sequence. It fills the
    annotations for frames from previous ones if the annotation for the current
    frame is not valid.

    Parameters
    ----------
    annotations : OrderedDict
        A dict of dict where the keys to the first dict are frame indices as
        strings (starting from 0). The inside dicts contain annotations for that
        frame. The dictionary needs to be an ordered dict in order for this to
        work.
    validator : ``callable``
        Takes a dict (annotations) and returns True if the annotations are valid.
        This can be a check based on minimal face size for example: see
        :any:`bob.bio.face.annotator.min_face_size_validator`.
    max_age : :obj:`int`, optional
        An integer indicating for a how many frames a detected face is valid if
        no detection occurs after such frame. A value of -1 == forever

    Yields
    ------
    str
        The index of frame.
    dict
        The corrected annotations of the frame.
    """
    # the annotations for the current frame
    current = None
    age = 0

    for k, annot in annotations.items():
        if validator(annot):
            current = annot
            age = 0
        elif max_age < 0 or age < max_age:
            age += 1
        else:  # no detections and age is larger than maximum allowed
            current = None

        yield k, current


class Base(bob.bio.base.annotator.Annotator):
    """The base class for video annotators."""

    @staticmethod
    def frame_ids_and_frames(frames):
        """Takes the frames and yields frame_ids and frames.

        Parameters
        ----------
        frames : :any:`bob.bio.video.VideoLikeContainer` or :any:`bob.bio.video.VideoAsArray` or :any:`numpy.array`
            The frames of the video file.

        Yields
        ------
        frame_id : str
            A string that represents the frame id.
        frame : :any:`numpy.array`
            The frame of the video file as an array.
        """
        if isinstance(frames, (utils.VideoAsArray, utils.VideoLikeContainer)):
            for fid, fr in zip(frames.indices, frames):
                yield fid, fr
        else:
            for fid, fr in enumerate(frames):
                yield str(fid), fr

    def annotate(self, frames, **kwargs):
        """Annotates videos.

        Parameters
        ----------
        frames : :any:`bob.bio.video.VideoLikeContainer` or :any:`bob.bio.video.VideoAsArray` or :any:`numpy.array`
            The frames of the video file.
        **kwargs
            Extra arguments that annotators may need.

        Returns
        -------
        OrderedDict
            A dictionary where its key is the frame id as a string and its value
            is a dictionary that are the annotations for that frame.


        .. note::

            You can use the :any:`Base.frame_ids_and_frames` functions to normalize
            the input in your implementation.
        """
        raise NotImplementedError()

    def transform(self, samples):
        """Takes a batch of data and annotates them.

        Each ``kwargs`` value is a list of parameters, with each element of those
        lists corresponding to each element of ``samples`` (for example:
        with ``[s1, s2, ...]`` as ``samples``, ``kwargs['annotations']``
        should contain ``[{<s1_annotations>}, {<s2_annotations>}, ...]``).
        """
        return [self.annotate(sample) for sample in samples]


class FailSafeVideo(Base):
    """A fail-safe video annotator.
    It tries several annotators in order and tries the next one if the previous
    one fails. However, the difference between this annotator and
    :any:`bob.bio.base.annotator.FailSafe` is that this one tries to use
    annotations from older frames (if valid) before trying the next annotator.

    .. warning::

        You must be careful in using this annotator since different annotators
        could have different results. For example the bounding box of one
        annotator be totally different from another annotator.

    Parameters
    ----------
    annotators : list
        A list of annotators to try.
    max_age : int
        The maximum number of frames that an annotation is valid for next frames.
        This value should be positive. If you want to set max_age to infinite,
        then you can use the :any:`bob.bio.video.annotator.Wrapper` instead.
    validator : ``callable``
        A function that takes the annotations of a frame and validates it.


    Please see :any:`Base` for more accepted parameters.
    """

    def __init__(
        self,
        annotators,
        max_age=15,
        validator=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if max_age <= 0:
            raise ValueError(
                f"max_age: `{max_age}' cannot be less than 1, If you want to set max_age to infinite,"
                "then you can use the :any:`bob.bio.video.annotator.Wrapper` with `normalize` set to True."
            )
        self.max_age = max_age

        if validator is None:
            validator = bob.bio.face.annotator.min_face_size_validator
        self.validator = validator

        self.annotators = []
        for annotator in annotators:
            if isinstance(annotator, str):
                annotator = bob.bio.base.load_resource(annotator, "annotator")
            self.annotators.append(annotator)

    def annotate(self, frames):
        """See :any:`Base.annotate`"""
        video_annotations = collections.OrderedDict()
        current = None
        age = 0
        for i, frame in self.frame_ids_and_frames(frames):
            for annotator in self.annotators:
                annot = annotator.transform([frame])[0]
                if annot and self.validator(annot):
                    current = annot
                    age = 0
                    break
                elif age < self.max_age:
                    age += 1
                    break
                else:  # no detections and age is larger than maximum allowed
                    current = None

                if current is not annot:
                    logger.debug("Annotator `%s' failed.", annotator)

            video_annotations[i] = current
        return video_annotations


class Wrapper(Base):
    """Annotates video files using the provided image annotator.
    See the documentation of :any:`Base` too.

    Parameters
    ----------
    annotator : :any:`bob.bio.base.annotator.Annotator` or str
        The image annotator to be used. The annotator could also be the name of a
        bob.bio.annotator resource which will be loaded.
    max_age : int
        see :any:`normalize_annotations`.
    normalize : bool
        If True, it will normalize annotations using :any:`normalize_annotations`
    validator : object
        See :any:`normalize_annotations` and
        :any:`bob.bio.face.annotator.min_face_size_validator` for one example.


    Please see :any:`Base` for more accepted parameters.

    .. warning::

        You should only set ``normalize`` to True only if you are annotating
        **all** frames of the video file.

    """

    def __init__(
        self,
        annotator,
        normalize=False,
        validator=None,
        max_age=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # load annotator configuration
        if isinstance(annotator, str):
            annotator = bob.bio.base.load_resource(annotator, "annotator")
        self.annotator = annotator

        if validator is None:
            validator = bob.bio.face.annotator.min_face_size_validator
        self.validator = validator

        self.normalize = normalize
        self.max_age = max_age

    def annotate(self, frames):
        """See :any:`Base.annotate`"""
        annotations = collections.OrderedDict()
        for i, frame in self.frame_ids_and_frames(frames):
            logger.debug("Annotating frame %s", i)
            annotations[i] = self.annotator.transform([frame])[0]
        if self.normalize:
            annotations = collections.OrderedDict(
                normalize_annotations(annotations, self.validator, self.max_age)
            )
        return annotations
