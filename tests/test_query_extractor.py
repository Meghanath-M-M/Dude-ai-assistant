from nova_agent.core.query_extractor import extract_project_name, extract_search_query


def test_extract_search_query_uses_the_longest_trigger():
    assert extract_search_query("search the web for rust ownership") == "rust ownership"
    assert extract_search_query("google this python asyncio") == "python asyncio"
    assert extract_search_query("look up sqlite wal mode") == "sqlite wal mode"
    assert extract_search_query("search rust ownership") == "rust ownership"


def test_extract_search_query_falls_back_to_filler_stripping():
    assert extract_search_query("open source vector databases") == "open source vector databases"
    assert extract_search_query("could you find chroma db") == "find chroma db"


def test_extract_search_query_is_empty_when_only_the_trigger_was_spoken():
    assert extract_search_query("search the web for") == ""
    assert extract_search_query("search") == ""
    assert extract_search_query("") == ""


def test_extract_project_name_returns_the_specific_name():
    assert extract_project_name("open my ml folder") == "ml"
    assert extract_project_name("open my ml project") == "ml"
    assert extract_project_name("launch the nova agent repo") == "nova agent"


def test_extract_project_name_is_empty_for_generic_requests():
    assert extract_project_name("open my project") == ""
    assert extract_project_name("launch my workspace") == ""
    assert extract_project_name("open the project folder") == ""
