from sklearn.base import BaseEstimator, TransformerMixin

from bob.bio.video import VideoLikeContainer
from bob.bio.video.transformer import VideoWrapper


class DummyEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, fail=False, **kwargs):
        super().__init__(**kwargs)
        self.fail = fail

    def transform(self, video, annotations=None):
        if self.fail:
            ret = [v if i % 2 == 1 else None for i, v in enumerate(video)]
            return ret
        return list(video)


def test_video_wrapper():
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    video_container = VideoLikeContainer(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], indices=indices
    )
    failed_container = VideoLikeContainer(
        [None, 2, None, 4, None, 6, None, 8, None, 10], indices=indices
    )
    failed_container2 = VideoLikeContainer(
        [None, None, None, 4, None, None, None, 8, None, None], indices=indices
    )

    for inputs, fail, oracle, kw in [
        ([video_container], False, video_container, dict()),
        ([video_container], True, failed_container, dict()),
        # because we are passing [video_container] to wrapper.transform, the
        # annotations must also be like [None]
        ([video_container], False, video_container, dict(annotations=[None])),
        ([video_container], True, failed_container, dict(annotations=[None])),
        ([failed_container], False, failed_container, dict()),
        ([failed_container], False, failed_container, dict(annotations=[None])),
        ([failed_container], True, failed_container2, dict()),
        ([failed_container], True, failed_container2, dict(annotations=[None])),
    ]:
        estimator = DummyEstimator(fail=fail)
        wrapper = VideoWrapper(estimator)
        assert wrapper.transform(inputs, **kw)[0] == oracle
