"""
taxo_extractor.py  —  BioTrace v5.3 Add-on
────────────────────────────────────────────────────────────────────────────
Standalone taxonomic entity extractor using BioBERT/TaxoNERD.
Designed to handle parenthetical subgenera and taxonomic headers.
"""
import logging

logger = logging.getLogger("biotrace.taxo_ner")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    logger.warning("Please run: pip install torch transformers")

class TaxoNERD:
    def __init__(self, model_name="nleguillar/taxoNERD", confidence_threshold=0.80):
        self.confidence_threshold = confidence_threshold
        self.pipeline = None
        
        if not _HF_AVAILABLE:
            logger.error("Hugging Face libraries missing. Cannot load TaxoNERD.")
            return

        logger.info(f"Loading TaxoNERD model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # aggregation_strategy="simple" automatically stitches sub-tokens back together
            # e.g., "Pleurobranchus", "(", "Susania", ")" -> "Pleurobranchus (Susania)"
            self.pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                aggregation_strategy="simple"
            )
            logger.info("TaxoNERD loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load TaxoNERD: {e}")

    def extract_organisms(self, text: str) -> list[str]:
        """Extracts scientific names from raw chunk text."""
        if not text or not self.pipeline:
            return []

        try:
            extracted = self.pipeline(text)
            species_set = set() # Automatically deduplicates
            
            for entity in extracted:
                # LIVB = Living Being (the label TaxoNERD uses for species/taxa)
                if entity['entity_group'] == 'LIVB' and entity['score'] >= self.confidence_threshold:
                    clean_name = entity['word'].strip()
                    
                    # Drop tiny generic abbreviations or isolated letters
                    if len(clean_name) > 3:
                        species_set.add(clean_name)
            
            return list(species_set)
        except Exception as e:
            logger.error(f"TaxoNERD extraction error: {e}")
            return []