import os

import bob.io.base

from bob.bio.base.database import ZTBioDatabase
from bob.bio.base.test.dummy.database import database as ATNT_DATABASE
from bob.bio.video import VideoLikeContainer
from bob.bio.video.database import VideoBioFile


class DummyBioFile(VideoBioFile):
    def load(self):
        file_name = self.make_path(
            self.original_directory, self.original_extension
        )
        fc = VideoLikeContainer(
            [bob.io.base.load(file_name)], [os.path.basename(file_name)]
        )
        return fc


class DummyDatabase(ZTBioDatabase):
    def __init__(self):
        # call base class constructor with useful parameters
        super(DummyDatabase, self).__init__(
            name="test",
            original_directory=ATNT_DATABASE.original_directory,
            original_extension=".pgm",
            check_original_files_for_existence=True,
            training_depends_on_protocol=False,
            models_depend_on_protocol=False,
        )
        self._db = ATNT_DATABASE

    def _make_bio(self, files):
        return [
            DummyBioFile(client_id=f.client_id, path=f.path, file_id=f.id)
            for f in files
        ]

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self._db.model_ids(groups, protocol)

    def objects(
        self,
        groups=None,
        protocol=None,
        purposes=None,
        model_ids=None,
        **kwargs
    ):
        return self._make_bio(
            self._db.objects(model_ids, groups, purposes, protocol, **kwargs)
        )

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        return []

    def zobjects(self, groups=None, protocol=None, **kwargs):
        return []

    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        return self._db.model_ids(groups)

    def t_enroll_files(self, t_model_id, group="dev"):
        return self.enroll_files(t_model_id, group)

    def z_probe_files(self, group="dev"):
        return self.probe_files(None, group)

    def file_names(self, files, directory, extension):
        if isinstance(files[0], list):
            files = list(list(zip(*files))[0])
        return super(DummyDatabase, self).file_names(
            files, directory, extension
        )

    def annotations(self, file):
        return None


database = DummyDatabase()
