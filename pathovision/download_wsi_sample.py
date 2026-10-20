"""Download and verify public OpenSlide sample slides."""
from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
from pathlib import Path
from urllib.request import urlopen

SAMPLE_URL = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"
SAMPLE_SHA256 = "ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7"
SAMPLE_PATH = Path("data/wsi/CMU-1-Small-Region.svs")
MULTILEVEL_ARCHIVE_URL = "https://github.com/WSILabs/wsi-fixtures/releases/download/v8/svs.tar"
MULTILEVEL_MEMBER = "svs/239551.svs"
MULTILEVEL_SHA256 = "019946dbe04eefaee18a90bb3222ae5ac51f081736ff1e195d0a52169ccc9535"
MULTILEVEL_SIZE = 26_637_925
MULTILEVEL_PATH = Path("data/wsi/239551.svs")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sample(destination: Path = SAMPLE_PATH) -> Path:
    """Download the public test slide, verify its pinned digest, and keep it local."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) == SAMPLE_SHA256:
            return destination
        raise ValueError(f"Existing sample does not match the expected SHA-256: {destination}")

    temporary = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    try:
        with urlopen(SAMPLE_URL, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != SAMPLE_SHA256:
            raise ValueError("Downloaded sample SHA-256 did not match the expected value")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def download_multilevel_sample(destination: Path = MULTILEVEL_PATH) -> Path:
    """Stream one CC-BY WSI from the pinned v8 fixture archive and verify its digest."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) == MULTILEVEL_SHA256:
            return destination
        raise ValueError(f"Existing multi-level sample does not match expected SHA-256: {destination}")

    temporary = destination.with_name(destination.name + ".part")
    found = False
    try:
        with urlopen(MULTILEVEL_ARCHIVE_URL, timeout=120) as response:
            with tarfile.open(fileobj=response, mode="r|*") as archive:
                for member in archive:
                    if member.name != MULTILEVEL_MEMBER:
                        continue
                    if member.size != MULTILEVEL_SIZE:
                        raise ValueError("Multi-level sample size did not match the pinned fixture")
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError("Could not read the multi-level slide from the fixture archive")
                    digest = hashlib.sha256()
                    with source, temporary.open("wb") as output:
                        while block := source.read(1024 * 1024):
                            digest.update(block)
                            output.write(block)
                    if digest.hexdigest() != MULTILEVEL_SHA256:
                        raise ValueError("Multi-level sample SHA-256 did not match the pinned value")
                    os.replace(temporary, destination)
                    found = True
                    break
        if not found:
            raise FileNotFoundError(f"Pinned fixture archive did not contain {MULTILEVEL_MEMBER}")
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-level", action="store_true", help="also download the 3-level CC BY 4.0 Aperio sample (about 64 MB streamed archive)")
    args = parser.parse_args()
    print(f"Verified OpenSlide sample: {download_sample()}")
    if args.multi_level:
        print(f"Verified multi-level Aperio sample: {download_multilevel_sample()}")
