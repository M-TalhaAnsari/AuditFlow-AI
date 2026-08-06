
import pytest
from src.generate import extract_json as generate_extract_json
from src.verify import extract_json as verify_extract_json

EXTRACTORS = [generate_extract_json, verify_extract_json]


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_extracts_clean_json(extractor):
    result = extractor('{"key": "value"}')
    assert result == {"key": "value"}


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_strips_markdown_fences_with_json_tag(extractor):
    result = extractor('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_strips_bare_markdown_fences(extractor):
    result = extractor('```\n{"key": "value"}\n```')
    assert result == {"key": "value"}


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_extracts_json_with_leading_stray_text(extractor):
    result = extractor('Sure, here is the JSON:\n{"key": "value"}')
    assert result == {"key": "value"}


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_extracts_json_with_trailing_stray_text(extractor):
    result = extractor('{"key": "value"}\nLet me know if you need anything else.')
    assert result == {"key": "value"}


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_raises_on_genuinely_invalid_json(extractor):
    with pytest.raises(Exception):
        extractor("this is not json at all")