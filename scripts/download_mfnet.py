"""Download the official public MFNet PNGs, preserving Drive resource keys.

Manifest contains official public file IDs, resource keys, byte sizes and MD5s.
It never contains account credentials. Existing verified files are reused.
Usage: python scripts/download_mfnet.py --destination datasets/ir_seg_dataset
Supports standard HTTPS_PROXY for machines that require an HTTP proxy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    hasher = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def download(item, destination):
    path = destination / item['folder'] / item['name']
    if path.is_file() and checksum(path) == item['md5Checksum']:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = {'export': 'download', 'id': item['id']}
    if item.get('resourceKey'):
        parameters['resourcekey'] = item['resourceKey']
    # This is the public download destination used by Drive, without the extra /uc redirect.
    url = 'https://drive.usercontent.google.com/download?' + urlencode(parameters)
    for attempt in range(3):
        try:
            temporary = path.with_suffix(path.suffix + '.part')
            with urlopen(url, timeout=45) as response, temporary.open('wb') as output:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(block)
            if temporary.stat().st_size != int(item['size']) or checksum(temporary) != item['md5Checksum']:
                raise ValueError(f'Checksum mismatch (possibly an HTML response): {path.name}')
            temporary.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/mfnet_download_manifest.json')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    files = manifest['files']
    for item in files:
        if item['folder'] not in ('', 'images', 'labels') or Path(item['name']).name != item['name']:
            raise ValueError('Invalid manifest path')
    if not 1 <= args.workers <= 32:
        raise ValueError('workers must be between 1 and 32')
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download, item, args.destination): item for item in files}
        for index, future in enumerate(as_completed(pending), 1):
            try:
                future.result()
            except Exception as error:
                errors.append({'file': pending[future]['name'], 'error': str(error)})
            if index % 100 == 0 or index == len(files):
                print(f'{index}/{len(files)} checked; failures={len(errors)}', flush=True)
    if errors:
        print(json.dumps(errors, indent=2))
        raise SystemExit(1)
    (args.destination / 'download_provenance.json').write_text(json.dumps({
        'source': manifest['source'], 'folder': manifest['folder'],
        'verified_files': len(files), 'checksum': 'All files match official Drive MD5 and byte size'}, indent=2))
    print(f'Official MFNet dataset verified: {args.destination.resolve()}', flush=True)


if __name__ == '__main__':
    main()
