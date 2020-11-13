"""
  YOUTUBE database implementation of bob.bio.base.database.ZTDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to YOUTUBE database, for
  verification experiments (good to use in bob.bio.base framework).
"""


import os

import bob.io.base
from bob.bio.base.database import ZTBioDatabase
from bob.extension import rc

from ..utils import VideoLikeContainer
from .database import VideoBioFile


class YoutubeBioFile(VideoBioFile):
    def __init__(self, f, **kwargs):
        super().__init__(client_id=f.client_id, path=f.path, file_id=f.id, **kwargs)
        if self.selection_style != "all":
            raise ValueError("Only selection style of 'all' is supported.")
        self._f = f

    def files(self, directory=None, extension=".jpg"):
        base_dir = self.make_path(directory, "")
        # collect all files from the data directory
        files = [os.path.join(base_dir, f) for f in sorted(os.listdir(base_dir))]
        # filter files with the given extension
        if extension is not None:
            files = [f for f in files if os.path.splitext(f)[1] == extension]
        return files

    def load(self, directory=None, extension=None, frame_selector=FrameSelector()):
        if extension not in (None, ".jpg"):
            raise ValueError(f"Unsupported extension {extension}")
        data, indices = [], []
        files = self.files(directory, extension)
        for f in frame_selector(files):
            file_name = os.path.join(self.make_path(directory, ""), f[0])
            indices.append(os.path.basename(file_name))
            data.append(bob.io.base.load(file_name))
        return VideoLikeContainer(data=data, indices=indices)


class YoutubeBioDatabase(ZTBioDatabase):
    """
    YouTube Faces database implementation of :py:class:`bob.bio.base.database.ZTBioDatabase` interface.
    It is an extension of an SQL-based database interface, which directly talks to :py:class:`bob.db.youtube.Database` database, for
    verification experiments (good to use in ``bob.bio`` framework).
    """

    def __init__(
        self,
        original_directory=rc["bob.db.youtube.directory"],
        original_extension=".jpg",
        annotation_extension=".labeled_faces.txt",
        **kwargs,
    ):
        from bob.db.youtube.query import Database as LowLevelDatabase

        self._db = LowLevelDatabase(
            original_directory, original_extension, annotation_extension
        )

        # call base class constructors to open a session to the database
        super(YoutubeBioDatabase, self).__init__(
            name="youtube",
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_extension=annotation_extension,
            **kwargs,
        )

    @property
    def original_directory(self):
        return self._db.original_directory

    @original_directory.setter
    def original_directory(self, value):
        self._db.original_directory = value

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self._db.model_ids(groups=groups, protocol=protocol)

    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        return self._db.tmodel_ids(protocol=protocol, groups=groups, **kwargs)

    def objects(
        self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs
    ):
        retval = self._db.objects(
            groups=groups,
            protocol=protocol,
            purposes=purposes,
            model_ids=model_ids,
            **kwargs,
        )
        return [YoutubeBioFile(f) for f in retval]

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        retval = self._db.tobjects(
            groups=groups, protocol=protocol, model_ids=model_ids, **kwargs
        )
        return [YoutubeBioFile(f) for f in retval]

    def zobjects(self, groups=None, protocol=None, **kwargs):
        retval = self._db.zobjects(groups=groups, protocol=protocol, **kwargs)
        return [YoutubeBioFile(f) for f in retval]

    def annotations(self, myfile):
        return self._db.annotations(myfile._f)

    def client_id_from_model_id(self, model_id, group="dev"):
        return self._db.get_client_id_from_file_id(model_id)
