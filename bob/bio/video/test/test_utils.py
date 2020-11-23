from bob.io.base.test_utils import datafile
import numpy as np
import bob.bio.video

regenerate_refs = False


def test_video_as_array():
    path = datafile("testvideo.avi", "bob.bio.video.test")

    video = bob.bio.video.VideoAsArray(path, selection_style="all")
    assert len(video) == 83, len(video)
    assert video.indices == range(83), video.indices

    video = bob.bio.video.VideoAsArray(path, selection_style="spread", max_number_of_frames=3)
    assert len(video) == 3, len(video)
    assert video.indices == [13, 41, 69], video.indices


def test_video_like_container():
    path = datafile("testvideo.avi", "bob.bio.video.test")

    video = bob.bio.video.VideoAsArray(path, selection_style="spread", max_number_of_frames=3)
    container = bob.bio.video.VideoLikeContainer(video, video.indices)

    container_path = datafile("video_like.hdf5", "bob.bio.video.test")

    if regenerate_refs:
        container.save(container_path)

    loaded_container = bob.bio.video.VideoLikeContainer.load(container_path)

    np.testing.assert_equal(np.array(container.data), np.array(loaded_container.data))
    np.testing.assert_equal(np.array(container.indices), np.array(loaded_container.indices))
