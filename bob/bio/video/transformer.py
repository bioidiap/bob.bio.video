import logging

from sklearn.base import BaseEstimator, TransformerMixin

from . import utils

logger = logging.getLogger(__name__)


class VideoWrapper(TransformerMixin, BaseEstimator):
    """Wrapper class to run image preprocessing algorithms on video data.

    **Parameters:**

    estimator : str or ``sklearn.base.BaseEstimator`` instance
      The transformer to be used to preprocess the frames.
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
        tags["bob_features_save_fn"] = utils.VideoLikeContainer.save_function
        tags["bob_features_load_fn"] = utils.VideoLikeContainer.load
        return tags

    def fit(self, X, y=None, **fit_params):
        """Does nothing"""
        return self
