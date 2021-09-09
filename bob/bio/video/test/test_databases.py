from nose.plugins.skip import SkipTest

import bob.bio.base
from bob.bio.face.test.test_databases import _check_annotations
import pkg_resources


def test_new_youtube():
    from bob.bio.video.database import YoutubeDatabase

    for protocol in [f"fold{i}" for i in range(10)]:

        database = YoutubeDatabase("fold0")
        references = database.references()
        probes = database.probes()

        assert len(references) == 500
        assert len(probes) == 500

