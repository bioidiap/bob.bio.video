import tempfile
import time
import pickle

import bob.bio.video
import numpy as np
from bob.bio.base.test.utils import is_library_available
from bob.io.base.test_utils import datafile
import imageio
from bob.io.image import to_bob

regenerate_refs = False


def test_video_as_array():
    path = datafile("testvideo.avi", "bob.bio.video.test")

    video = bob.bio.video.VideoAsArray(path, selection_style="all")
    assert len(video) == 83, len(video)
    assert video.indices == range(83), video.indices
    assert video.shape == (83, 3, 480, 640), video.shape
    np.testing.assert_equal(video[0][:, 0, 0], np.array([78, 103, 100]))

    video_slice = video[1:2, 1:-1, 1:-1, 1:-1]
    assert video_slice.shape == (1, 1, 478, 638), video_slice.shape

    # test the slice against the video loaded by imageio directly
    video = to_bob(imageio.get_reader(path).get_data(1))
    video = video[1:-1, 1:-1, 1:-1]
    video = video[None, ...]
    np.testing.assert_allclose(video, video_slice)

    video = bob.bio.video.VideoAsArray(path, max_number_of_frames=3)
    assert len(video) == 3, len(video)
    assert video.indices == [13, 41, 69], video.indices
    assert video.shape == (3, 3, 480, 640), video.shape
    np.testing.assert_equal(video[-1][:, 0, 0], np.array([75, 100, 97]))

    # pickle video and unpickle to see if it works
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        pickle.dump(video, f)
        f.seek(0)
        pickle.load(f)


@is_library_available("dask")
def test_video_as_array_vs_dask():
    import dask

    path = datafile("testvideo.avi", "bob.bio.video.test")
    start = time.time()
    video = bob.bio.video.VideoAsArray(path, selection_style="all")
    video = dask.array.from_array(video, (20, 1, 480, 640))
    video = video.compute(scheduler="single-threaded")
    load_time = time.time() - start

    start = time.time()
    reference = to_bob(np.array(list((imageio.get_reader(path).iter_data()))))
    load_time2 = time.time() - start
    # Here, we're also chunking each frame, but normally we would only chunk the first axis.
    print(
        f"FYI: It took {load_time:.2f} s to load the video with dask and {load_time2:.2f} s "
        "to load directly. The slower loading with dask is expected."
    )
    np.testing.assert_allclose(reference, video)


def test_video_like_container():
    path = datafile("testvideo.avi", "bob.bio.video.test")

    video = bob.bio.video.VideoAsArray(
        path, selection_style="spread", max_number_of_frames=3
    )
    container = bob.bio.video.VideoLikeContainer(video, video.indices)

    container_path = datafile("video_like.hdf5", "bob.bio.video.test")

    if regenerate_refs:
        container.save(container_path)

    loaded_container = bob.bio.video.VideoLikeContainer.load(container_path)
    assert container == loaded_container

    # test saving and loading None arrays
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        data = [None] * 10 + [1]
        indices = range(11)
        frame_container = bob.bio.video.VideoLikeContainer(data, indices)
        frame_container.save(f.name)

        loaded = bob.bio.video.VideoLikeContainer.load(f.name)
        assert loaded == frame_container
