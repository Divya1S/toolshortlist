"""Tests for the matching rules used by the overlap audit."""

import json

from toolshortlist.audit import ToolIndex, jaccard, normalise, parse_tool


def tool(name, description, **extra):
    return parse_tool(json.dumps({"name": name, "description": description, **extra}))


def test_normalise():
    assert normalise("  What's the WEATHER in Paris?? ") == "what s the weather in paris"


def test_jaccard():
    assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0
    assert jaccard({"a", "b", "c"}, {"a", "b", "d"}) == 0.5  # 2 shared out of 4 distinct
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard(set(), set()) == 0.0


def test_parse_tool_ignores_key_order():
    a = parse_tool('{"name": "Get Weather", "description": "Current weather."}')
    b = parse_tool('{"description": "Current weather.", "name": "Get Weather"}')
    assert a == b
    assert a[1:] == ("get weather", "current weather")


def test_parse_tool_without_name_or_json():
    assert parse_tool('{"func": "f"}')[1:] == ("", "")
    assert parse_tool("not json") == ("not json", "", "")


def test_tool_matching_criteria():
    long_description = "Returns the current weather for a city, including temperature and wind."
    index = ToolIndex()
    index.add(tool("get_weather", long_description, parameters={"city": "str"}))

    assert index.match(tool("get_weather", long_description, parameters={"city": "str"})) \
        == "identical documentation"
    assert index.match(tool("get_weather", long_description)) == "same name and description"
    assert index.match(tool("get_weather", long_description + " Fast")) \
        == "same name, near-identical description"
    assert index.match(tool("weather_now", long_description)) == "same description under another name"
    assert index.match(tool("get_weather", "Books a flight between two airports.")) is None
    assert index.match(tool("book_flight", "Books a flight between two airports.")) is None


def test_short_shared_description_is_not_a_match():
    index = ToolIndex()
    index.add(tool("search", "Search items"))
    assert index.match(tool("find", "Search items")) is None
