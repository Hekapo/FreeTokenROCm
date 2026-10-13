"""One bounded request at configured 64K capacity, with recorded streaming timings."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


def get_stats():
    with urllib.request.urlopen('http://127.0.0.1:1919/v1/stats', timeout=5) as response:
        return json.load(response)


def run(output: Path) -> dict:
    if output.exists():
        raise RuntimeError('Use a fresh smoke output file')
    initial = get_stats()
    if (initial['model']['ctx'] != 65536 or initial['kv']['total_pages'] != 65536
            or initial['kv']['page_size'] != 1):
        raise RuntimeError('The live context/KV capacity is not 65536')
    payload = {
        'model': 'qwen3.6-35b-a3b-q4_1',
        'messages': [{'role': 'user', 'content': 'Print 1 to 100, comma-separated.'}],
        'temperature': 0, 'max_tokens': 512, 'stream': True,
        'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': False},
    }
    request = urllib.request.Request('http://127.0.0.1:1919/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    result = {'configured_capacity': 65536, 'kv_capacity': 65536, 'request_count': 1,
              'stats_before': initial, 'content': '', 'reasoning': '', 'done': False,
              'usage': None, 'finish_reason': None, 'error': None, 'passed': False}
    started = time.perf_counter()
    first = last = None
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result['status_code'] = response.status
            for raw in response:
                now = time.perf_counter()
                if now - started > 45:
                    raise TimeoutError('Short smoke exceeded 45 seconds')
                if not raw.startswith(b'data:'):
                    continue
                data = raw[5:].strip()
                if data == b'[DONE]':
                    result['done'] = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    result['usage'] = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    content = delta.get('content') or ''
                    reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if content:
                        first = now if first is None else first
                        last = now
                    result['content'] += content
                    result['reasoning'] += reasoning
                    if choice.get('finish_reason'):
                        result['finish_reason'] = choice['finish_reason']
        result['stats_after'] = get_stats()
        result['content_sha256'] = hashlib.sha256(result['content'].encode()).hexdigest()
        result['passed'] = (result['done'] and result['finish_reason'] == 'stop'
                            and re.findall(r'\d+', result['content']) == [str(n) for n in range(1, 101)]
                            and result['reasoning'] == '' and result['usage'] is not None
                            and result['usage']['prompt_tokens'] == 24
                            and result['usage']['completion_tokens'] == 391
                            and result['content_sha256'] == '34a4f1e5bb080915a30b7f67a8546b8e72da130622436caa0fcb81a2eb62c0ee')
    except Exception as error:
        result['error'] = repr(error)
    finally:
        result['total_s'] = round(time.perf_counter() - started, 6)
        result['ttft_s'] = round(first - started, 6) if first is not None else None
        result['last_token_s'] = round(last - started, 6) if last is not None else None
        usage = result['usage'] or {}
        result['decode_tps'] = (round((usage.get('completion_tokens', 0) - 1) / (last - first), 3)
                                if first is not None and last is not None and last > first else None)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {'stats_before', 'stats_after', 'content', 'reasoning'}}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output.resolve())['passed'] else 2)
