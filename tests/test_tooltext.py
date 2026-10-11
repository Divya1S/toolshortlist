import json

import pytest

from toolshortlist.tooltext import fields, render_tool

TOOLBENCH = json.dumps({"category_name": "Weather", "required_parameters": [{"name": "city"}],
                        "optional_parameters": [], "method": "GET", "name": "Get_Forecast",
                        "description": "Five-day forecast for a city."})
GORILLA = json.dumps({"domain": "Vision", "api_call": "torch.hub.load('resnet50')",
                      "functionality": "Image classification", "description": "ResNet-50 from torchvision."})


def test_fields():
    assert fields(TOOLBENCH) == ("Get_Forecast", "Five-day forecast for a city.")
    assert fields(GORILLA) == ("torch.hub.load('resnet50')", "ResNet-50 from torchvision.")
    assert fields("not json") == ("", "")
    assert fields('{"parameters": {}}') == ("", "")


def test_render_modes():
    assert render_tool(TOOLBENCH, "full") == TOOLBENCH
    assert render_tool(TOOLBENCH, "name") == "Get_Forecast"
    assert render_tool(TOOLBENCH, "name_description") == "Get_Forecast: Five-day forecast for a city."


def test_render_falls_back_when_fields_are_missing():
    assert render_tool('{"description": "Only a description."}', "name") == "Only a description."
    assert render_tool('{"parameters": {}}', "name_description") == '{"parameters": {}}'
    assert render_tool("not json", "name") == "not json"


def test_unknown_mode():
    with pytest.raises(ValueError):
        render_tool(TOOLBENCH, "everything")
