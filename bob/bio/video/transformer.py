import logging

from sklearn.base import BaseEstimator, TransformerMixin

from . import utils

logger = logging.getLogger(__name__)


class VideoWrapper(TransformerMixin, BaseEstimator):
    """Wrapper class to run image preprocessing algorithms on video data.

    This class provides functionality to read original video data from several databases.
    So far, the video content from :ref:`bob.db.mobio <bob.db.mobio>` and the image list content from :ref:`bob.db.youtube <bob.db.youtube>` are supported.

    Furthermore, frames are extracted from these video data, and a ``preprocessor`` algorithm is applied on all selected frames.
    The preprocessor can either be provided as a registered resource, i.e., one of :ref:`bob.bio.face.preprocessors`, or an instance of a preprocessing class.
    Since most of the databases do not provide annotations for all frames of the videos, commonly the preprocessor needs to apply face detection.

    The ``frame_selector`` can be chosen to select some frames from the video.
    By default, a few frames spread over the whole video sequence are selected.

    The ``quality_function`` is used to assess the quality of the frame.
    If no ``quality_function`` is given, the quality is based on the face detector, or simply left as ``None``.
    So far, the quality of the frames are not used, but it is foreseen to select frames based on quality.

    **Parameters:**

    preprocessor : str or :py:class:`bob.bio.base.preprocessor.Preprocessor` instance
      The preprocessor to be used to preprocess the frames.

    frame_selector : :py:class:`bob.bio.video.FrameSelector`
      A frame selector class to define, which frames of the video to use.

    quality_function : function or ``None``
      A function assessing the quality of the preprocessed image.
      If ``None``, no quality assessment is performed.
      If the preprocessor contains a ``quality`` attribute, this is taken instead.

    compressed_io : bool
      Use compression to write the resulting preprocessed HDF5 files.
      This is experimental and might cause trouble.
      Use this flag with care.

    read_original_data: callable or ``None``
       Function that loads the raw data.
       If not explicitly defined the raw data will be loaded by :py:meth:`bob.bio.video.database.VideoBioFile.load`
       using the specified ``frame_selector``

    """

    def __init__(
        self,
        estimator,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.estimator = estimator

    def transform(self, videos, **kwargs):
        transformed_videos = []
        for i, video in enumerate(videos):

            if not hasattr(video, "indices"):
                raise ValueError(
                    f"The input video: {video}\n does not have indices.\n "
                    f"Processing failed in {self}"
                )

            kw = {}
            if kwargs:
                kw = {k: v[i] for k, v in kwargs.items()}
            if "annotations" in kw:
                kw["annotations"] = [
                    kw["annotations"].get(index, kw["annotations"].get(str(index)))
                    for index in video.indices
                ]

            data = self.estimator.transform(video, **kw)

            dl, vl = len(data), len(video)
            if dl != vl:
                raise RuntimeError(
                    f"Length of transformed data ({dl}) using {self.estimator}"
                    f" is different from the length of input video: {vl}"
                )

            # handle None's
            indices = [idx for d, idx in zip(data, video.indices) if d is not None]
            data = [d for d in data if d is not None]

            data = utils.VideoLikeContainer(data, indices)
            transformed_videos.append(data)
        return transformed_videos

    def _more_tags(self):
        tags = self.estimator._get_tags()
        tags["bob_features_save_fn"] = utils.VideoLikeContainer.save
        tags["bob_features_load_fn"] = utils.VideoLikeContainer.load
        return tags

    def fit(self, X, y=None, **fit_params):
        """Does nothing"""
        return self
