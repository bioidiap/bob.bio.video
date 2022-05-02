def test_new_youtube():
    from bob.bio.video.database import YoutubeDatabase

    for protocol in [f"fold{i}" for i in range(10)]:

        database = YoutubeDatabase("fold0")
        references = database.references()
        probes = database.probes()

        assert len(references) == 500
        assert len(probes) == 500
