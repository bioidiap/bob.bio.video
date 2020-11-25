from nose.plugins.skip import SkipTest

import bob.bio.base
from bob.bio.base.test.utils import db_available
from bob.bio.base.test.test_database_implementations import check_database_zt
from bob.bio.face.test.test_databases import _check_annotations
import pkg_resources


@db_available("youtube")
def test_youtube():
    database = bob.bio.base.load_resource(
        "youtube", "database", preferred_package="bob.bio.video"
    )
    try:
        check_database_zt(database, training_depends=True, models_depend=True)
    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e
        )
    try:
        if database.database.original_directory is None:
            raise SkipTest("The annotations cannot be queried as original_directory is None")
        _check_annotations(database, limit_files=1000, topleft=True, framed=True)
    except IOError as e:
        raise SkipTest(
            "The annotations could not be queried; probably the annotation files are missing. Here is the error: '%s'"
            % e
        )


@db_available("youtube")
def test_youtube_load_method():
    database = bob.bio.base.load_resource(
        "youtube", "database", preferred_package="bob.bio.video"
    )
    database.database.original_directory = pkg_resources.resource_filename(
        "bob.bio.video", "test/data"
    )
    youtube_db_sample = [
        sample
        for sample_set in database.references(group="dev")
        for sample in sample_set
        if sample.key == "Aaron_Eckhart/0"
    ][0]

    frame_container = youtube_db_sample.data

    assert len(frame_container) == 2
