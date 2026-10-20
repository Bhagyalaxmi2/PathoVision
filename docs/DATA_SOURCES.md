# Data sources and attribution

## OpenSlide CMU small-region example

- **File:** `data/wsi/CMU-1-Small-Region.svs`
- **Source:** [OpenSlide test data](https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs)
- **License:** CC0 1.0
- **Use in PathoVision:** exercises SVS reading and metadata handling. It is a small, single-level, unlabeled example and is not model-evaluation data.
- **SHA-256:** `ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7`

## Aperio 239551 multilevel example

- **File:** `data/wsi/239551.svs`
- **Source:** [WSILabs wsi-fixtures, pinned v8 release](https://github.com/WSILabs/wsi-fixtures/releases/tag/v8)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Required attribution:** “Whole-slide image '239551', © Toby C. Cornish / WSILabs, licensed under CC BY 4.0.”
- **Use in PathoVision:** 23,999 × 8,015 pixels, Aperio SVS, 20×, about 0.4967 µm/pixel, three pyramid levels. It exercises multilevel thumbnail and level-0 region reads; it is unlabeled and is not model-evaluation data.
- **SHA-256:** `019946dbe04eefaee18a90bb3222ae5ac51f081736ff1e195d0a52169ccc9535`
- **Changes:** PathoVision does not modify the source slide. It computes heuristic masks, tile measurements, and visual overlays at runtime.

The slide binaries remain under Git-ignored `data/`. The optional downloader pins the fixture release and checks the extracted slide checksum before saving it.
