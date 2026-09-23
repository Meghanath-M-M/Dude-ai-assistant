from nova_agent.core.query_extractor import (
    extract_app_phrase,
    extract_project_name,
    extract_search_query,
)


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


def test_extract_app_phrase_prefers_known_app_words():
    assert extract_app_phrase("open chrome") == "chrome"
    assert extract_app_phrase("launch visual studio code") == "visual studio code"
    assert extract_app_phrase("open vs code") == "vs code"


def test_extract_app_phrase_keeps_unknown_app_names_for_the_resolver():
    assert extract_app_phrase("open gallery") == "gallery"
    assert extract_app_phrase("start notepad") == "notepad"
    assert extract_app_phrase("switch to edge") == "edge"


def test_extract_app_phrase_is_none_without_a_launch_shape():
    assert extract_app_phrase("just talking about apps") is None
    assert extract_app_phrase("") is None
    assert extract_app_phrase("open") is None  # trigger with nothing after it
