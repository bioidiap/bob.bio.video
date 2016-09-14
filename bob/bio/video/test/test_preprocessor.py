#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
import numpy
import bob.io.base
import bob.io.image
import bob.io.video
import bob.bio.base
import bob.bio.video
import bob.db.base
import pkg_resources
from bob.bio.video.test.dummy.database import DummyBioFile

regenerate_refs = False

def test_annotations():
    original_path = pkg_resources.resource_filename("bob.bio.face.test", "")
    image_files = DummyBioFile(bob.bio.base.database.BioFile(client_id=1, file_id=1, path="data/testimage"))

    # use annotations to grep
    annotations = {os.path.basename(image_files.make_path(original_path, ".jpg")): bob.db.base.read_annotation_file(
        pkg_resources.resource_filename("bob.bio.face.test", "data/testimage.pos"), 'named')}

    # video preprocessor using a face crop preprocessor
    frame_selector = bob.bio.video.FrameSelector(selection_style="all")
    preprocessor = bob.bio.video.preprocessor.Wrapper('face-crop-eyes', frame_selector, compressed_io=False)

    # read original data
    original = preprocessor.read_original_data(image_files, original_path, ".jpg")

    assert isinstance(original, bob.bio.video.FrameContainer)
    assert len(original) == 1
    assert original[0][0] == os.path.basename(image_files.make_path(original_path, ".jpg"))

    # preprocess data including annotations
    preprocessed = preprocessor(original, annotations)
    assert isinstance(preprocessed, bob.bio.video.FrameContainer)
    assert len(preprocessed) == 1
    assert preprocessed[0][0] == os.path.basename(image_files.make_path(original_path, ".jpg"))
    assert preprocessed[0][2] is None
    assert numpy.allclose(preprocessed[0][1],
                          bob.io.base.load(pkg_resources.resource_filename("bob.bio.face.test", "data/cropped.hdf5")))

def test_detect():

    def read_original_data(video_object, original_path, original_extension):
        return frame_selector(video_object.make_path(original_path, original_extension))

    # load test video
    original_path = pkg_resources.resource_filename("bob.bio.video.test", "")
    video_object = bob.bio.video.database.VideoBioFile(bob.bio.base.database.BioFile(
                                                       client_id=1, file_id=1, path="data/testvideo"))

    frame_selector = bob.bio.video.FrameSelector(max_number_of_frames=3, selection_style="spread")
    preprocessor = bob.bio.video.preprocessor.Wrapper('face-detect', frame_selector, compressed_io=False,
                                                      read_original_data=read_original_data)

    video = preprocessor.read_original_data(video_object, original_path,".avi")
    assert isinstance(video, bob.bio.video.FrameContainer)

    preprocessed_video = preprocessor(video)
    assert isinstance(preprocessed_video, bob.bio.video.FrameContainer)

    reference_file = pkg_resources.resource_filename("bob.bio.video.test", "data/preprocessed.hdf5")
    if regenerate_refs:
        preprocessed_video.save(bob.io.base.HDF5File(reference_file, 'w'))
    reference_data = bob.bio.video.FrameContainer(bob.io.base.HDF5File(reference_file, 'r'))

    assert preprocessed_video.is_similar_to(reference_data)


def test_flandmark():

    def read_original_data(video_object, original_path, original_extension):
        return frame_selector(video_object.make_path(original_path, original_extension))

    original_path = pkg_resources.resource_filename("bob.bio.video.test", "")
    video_object = bob.bio.video.database.VideoBioFile(bob.bio.base.database.BioFile(
                                                       client_id=1, file_id=1, path="data/testvideo"))
    frame_selector = bob.bio.video.FrameSelector(max_number_of_frames=3, selection_style="spread")

    preprocessor = bob.bio.video.preprocessor.Wrapper('landmark-detect', frame_selector, compressed_io=False,
                                                      read_original_data=read_original_data)
    video = preprocessor.read_original_data(video_object, original_path, ".avi")
    assert isinstance(video, bob.bio.video.FrameContainer)

    preprocessed_video = preprocessor(video)
    assert isinstance(preprocessed_video, bob.bio.video.FrameContainer)

    reference_file = pkg_resources.resource_filename("bob.bio.video.test", "data/preprocessed-flandmark.hdf5")
    if regenerate_refs:
        preprocessed_video.save(bob.io.base.HDF5File(reference_file, 'w'))
    reference_data = bob.bio.video.FrameContainer(bob.io.base.HDF5File(reference_file, 'r'))

    assert preprocessed_video.is_similar_to(reference_data)
