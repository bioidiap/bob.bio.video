"""
  YOUTUBE database implementation of bob.bio.base.database.ZTDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to YOUTUBE database, for
  verification experiments (good to use in bob.bio.base framework).
"""


import os

import bob.io.base
from bob.bio.base.database import ZTBioDatabase
from bob.extension import rc

from ..utils import VideoLikeContainer, select_frames
from .database import VideoBioFile


class YoutubeBioFile(VideoBioFile):
    def __init__(self, f, **kwargs):
        super().__init__(client_id=f.client_id, path=f.path, file_id=f.id, **kwargs)
        self._f = f

    def files(self):
        base_dir = self.make_path(self.original_directory, "")
        # collect all files from the data directory
        files = [os.path.join(base_dir, f) for f in sorted(os.listdir(base_dir))]
        # filter files with the given extension
        if self.original_extension is not None:
            files = [
                f for f in files if os.path.splitext(f)[1] == self.original_extension
            ]
        return files

    def load(self, *args, **kwargs):
        files = self.files()
        files_indices = select_frames(
            len(files),
            max_number_of_frames=self.max_number_of_frames,
            selection_style=self.selection_style,
            step_size=self.step_size,
        )
        data, indices = [], []
        for i, file_name in enumerate(files):
            if i not in files_indices:
                continue
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

    def _populate_files_attrs(self, files):
        for f in files:
            f.original_directory = self.original_directory
            f.original_extension = self.original_extension
            f.annotation_extension = self.annotation_extension
        return files

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
        return self._populate_files_attrs([YoutubeBioFile(f) for f in retval])

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        retval = self._db.tobjects(
            groups=groups, protocol=protocol, model_ids=model_ids, **kwargs
        )
        return self._populate_files_attrs([YoutubeBioFile(f) for f in retval])

    def zobjects(self, groups=None, protocol=None, **kwargs):
        retval = self._db.zobjects(groups=groups, protocol=protocol, **kwargs)
        return self._populate_files_attrs([YoutubeBioFile(f) for f in retval])

    def annotations(self, myfile):
        return self._db.annotations(myfile._f)

    def client_id_from_model_id(self, model_id, group="dev"):
        return self._db.get_client_id_from_file_id(model_id)
