import collections
import os

import pkg_resources

import bob.bio.video
import bob.io.base

from bob.bio.face.test.test_annotators import _assert_mtcnn
from bob.bio.video.test.dummy.database import DummyBioFile


class FailSucessAnnotator(bob.bio.base.annotator.Annotator):
    """An annotator that fails for every second time it is called."""

    def __init__(self, **kwargs):
        super(FailSucessAnnotator, self).__init__(**kwargs)
        self.failed_last_time = True

    def annotate(self, image, **kwargs):
        if not self.failed_last_time:
            self.failed_last_time = True
            return None
        else:
            self.failed_last_time = False
            return {"topleft": (0, 0), "bottomright": (64, 64)}

    def transform(self, images):
        return [self.annotate(img) for img in images]


def test_wrapper():

    original_path = pkg_resources.resource_filename("bob.bio.face.test", "")
    image_files = DummyBioFile(
        client_id=1,
        file_id=1,
        path="data/testimage",
        original_directory=original_path,
        original_extension=".jpg",
    )
    # read original data
    original = image_files.load()

    # video preprocessor using a face crop preprocessor
    annotator = bob.bio.video.annotator.Wrapper("mtcnn")

    assert isinstance(original, bob.bio.video.VideoLikeContainer)
    assert len(original) == 1
    assert original.indices[0] == os.path.basename(
        image_files.make_path(original_path, ".jpg")
    )

    # annotate data
    annot = annotator.transform([original])[0]

    assert isinstance(annot, collections.OrderedDict), annot
    _assert_mtcnn(annot["testimage.jpg"])


def _get_test_video():
    original_path = pkg_resources.resource_filename("bob.bio.video.test", "")
    # here I am using 3 frames to test normalize but in real applications this
    # should not be done.
    video_object = bob.bio.video.database.VideoBioFile(
        client_id=1,
        file_id=1,
        path="data/testvideo",
        original_directory=original_path,
        original_extension=".avi",
        max_number_of_frames=3,
        selection_style="spread",
    )
    video = video_object.load()
    assert isinstance(video, bob.bio.video.VideoAsArray)
    return video


def test_wrapper_normalize():

    video = _get_test_video()

    annotator = bob.bio.video.annotator.Wrapper("mtcnn", normalize=True)

    annot = annotator.transform([video])[0]

    # check if annotations are ordered by frame number
    assert list(annot.keys()) == sorted(annot.keys(), key=int), annot


def test_failsafe_video():

    video = _get_test_video()

    annotator = bob.bio.video.annotator.FailSafeVideo(
        [FailSucessAnnotator(), "mtcnn"]
    )

    annot = annotator.transform(video)[0]

    # check if annotations are ordered by frame number
    assert list(annot.keys()) == sorted(annot.keys(), key=int), annot

    # check if the failsuccess annotator was used for all frames
    for _, annotations in annot.items():
        assert "topleft" in annotations, annot
        assert annotations["topleft"] == (0, 0), annot
        assert annotations["bottomright"] == (64, 64), annot
