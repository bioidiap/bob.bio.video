import logging

from sklearn.base import BaseEstimator, TransformerMixin

from bob.pipelines.wrappers import _check_n_input_output, _frmt

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
            if "annotations" in kw and kw["annotations"] is not None:
                kw["annotations"] = [
                    kw["annotations"].get(
                        index, kw["annotations"].get(str(index))
                    )
                    for index in video.indices
                ]

            # remove None's before calling and add them back in data later
            # Isolate invalid samples (when previous transformers returned None)
            invalid_ids = [i for i, frame in enumerate(video) if frame is None]
            valid_frames = [frame for frame in video if frame is not None]

            # remove invalid kw args as well
            for k, v in kw.items():
                if v is None:
                    continue
                kw[k] = [vv for j, vv in enumerate(v) if j not in invalid_ids]

            # Process only the valid samples
            output = None
            if len(valid_frames) > 0:
                output = self.estimator.transform(valid_frames, **kw)
                _check_n_input_output(
                    valid_frames, output, f"{_frmt(self.estimator)}.transform"
                )

            if output is None:
                output = [None] * len(valid_frames)

            # Rebuild the full batch of samples (include the previously failed)
            if len(invalid_ids) > 0:
                output = list(output)
                for j in invalid_ids:
                    output.insert(j, None)

            data = utils.VideoLikeContainer(output, video.indices)
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
