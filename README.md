Final runnable bundle for `biotrace531`.

Included here:
- Current `biotrace_v5.py` and `biotrace_v53.py`
- Supporting local BioTrace modules copied from the working checkout
- `biotrace_schema.py`, `biotrace_scientific_chunker.py`, `biotrace_wiki.py`
- `ner_model/`
- `biodiversity_data/`
- `requirements.txt`
- `environment.yml`

Notes:
- The original `ModuleNotFoundError` for `biotrace_geocoding_lifestage_patch` is fixed because the patch/helper modules are now present in this folder.
- `biotrace_ocr.py` was not copied because no source file exists in either local checkout. `biotrace_v5.py` already treats it as optional and degrades gracefully when it is absent.
