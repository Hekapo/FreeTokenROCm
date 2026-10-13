import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def client():
    path = Path(__file__).resolve().parents[2] / 'scripts/windows-rocm1010/smoke_request.py'
    spec = importlib.util.spec_from_file_location('rocm1010_smoke_client_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stream:
    status = 200

    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        for event in self.events:
            yield b'data: ' + json.dumps(event).encode() + b'\n'
        yield b'data: [DONE]\n'


def stats():
    return {'model': {'ctx': 65536}, 'kv': {'total_pages': 65536, 'page_size': 1}}


def test_one_request_validates_capacity_output_and_usage(client, monkeypatch, tmp_path):
    monkeypatch.setattr(client, 'get_stats', stats)
    content = ', '.join(map(str, range(1, 101)))
    requests = []

    def open_request(request, timeout):
        requests.append(json.loads(request.data))
        assert timeout == 45
        return Stream([{'choices': [{'delta': {'content': content}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 24, 'completion_tokens': 391}}])

    monkeypatch.setattr(client.urllib.request, 'urlopen', open_request)
    result = client.run(tmp_path / 'result.json')
    assert result['passed'] is True
    assert len(requests) == 1 and requests[0]['max_tokens'] == 512
    assert requests[0]['chat_template_kwargs'] == {'enable_thinking': False}


def test_capacity_rejection_happens_before_request(client, monkeypatch, tmp_path):
    value = stats()
    value['kv']['total_pages'] = 8192
    monkeypatch.setattr(client, 'get_stats', lambda: value)
    monkeypatch.setattr(client.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('request started'))
    with pytest.raises(RuntimeError, match='capacity'):
        client.run(tmp_path / 'result.json')


def test_wrong_answer_is_recorded_as_failure(client, monkeypatch, tmp_path):
    monkeypatch.setattr(client, 'get_stats', stats)
    monkeypatch.setattr(client.urllib.request, 'urlopen', lambda *a, **k: Stream([
        {'choices': [{'delta': {'content': '1, 2, 4'}, 'finish_reason': 'stop'}],
         'usage': {'prompt_tokens': 24, 'completion_tokens': 391}}]))
    assert client.run(tmp_path / 'result.json')['passed'] is False


def test_stream_error_is_saved(client, monkeypatch, tmp_path):
    monkeypatch.setattr(client, 'get_stats', stats)
    monkeypatch.setattr(client.urllib.request, 'urlopen', lambda *a, **k: Stream([
        {'error': {'message': 'backend stopped'}}]))
    path = tmp_path / 'result.json'
    result = client.run(path)
    assert result['passed'] is False and 'backend stopped' in result['error']
    assert json.loads(path.read_text())['passed'] is False
