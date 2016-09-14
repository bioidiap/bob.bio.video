#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Wed 20 July 14:43:22 CEST 2016

"""
  Verification API for bob.db.voxforge
"""

from bob.bio.base.database.file import BioFile
from bob.bio.video.utils.FrameSelector import FrameSelector


class VideoBioFile(BioFile):
    def __init__(self, f):
        """
        Initializes this File object with an File equivalent for
        VoxForge database.
        """
        super(VideoBioFile, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)

        self.__f = f

    def load(self, directory=None, extension='.avi'):
        return FrameSelector()(self.make_path(directory, extension))



