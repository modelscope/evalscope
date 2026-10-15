"""Serve a local ModelScope Clef release using its official PyTorch implementation."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    """Load the complete release, including its joint head, and expose System One."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--model-name', default='Cloudflare/clef-flash')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18131)
    args = parser.parse_args()
    if not args.model_path.is_dir():
        raise ValueError('Download the complete model from ModelScope before starting this server.')
    sys.path.insert(0, str(args.model_path.resolve()))
    from fastapi import FastAPI, HTTPException
    import uvicorn

    from joint_schema_model import load_release_model, systemone

    model, processor = load_release_model(args.model_path, attn_implementation='eager')
    release = args.model_path / 'evalscope_release_manifest.json'
    revision = json.loads(release.read_text())['revision'] if release.is_file() else None
    app = FastAPI()
    lock = asyncio.Lock()

    @app.get('/health')
    async def health() -> dict[str, str]:
        return {'status': 'ready', 'model': args.model_name, 'backend': 'official_pytorch'}

    @app.post('/v1/systemone')
    async def decide(body: dict[str, Any]) -> dict[str, Any]:
        if body.get('model') != args.model_name:
            raise HTTPException(400, 'Unknown model name.')
        try:
            async with lock:
                result = await asyncio.to_thread(systemone, model, processor, body)
                if revision:
                    result['model_revision'] = revision
                return result
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
