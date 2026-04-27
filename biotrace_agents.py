import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from biotrace_api_integrations import TaxonomicAPI, GeospatialAPI

logger = logging.getLogger("biotrace_agents")

# Pydantic models for structured output
class SamplingEvent(BaseModel):
    date: str = Field(default="Not Reported")
    depth_m: str = Field(default="Not Reported")
    method: str = Field(default="Not Reported")

class RawOccurrence(BaseModel):
    recorded_name: str = Field(description="Scientific name exactly as written")
    source_citation: str = Field(description="Citation or source document")
    habitat: str = Field(default="Not Reported")
    sampling_event: SamplingEvent = Field(default_factory=SamplingEvent)
    raw_text_evidence: str = Field(description="Exact verbatim sentence")
    verbatim_locality: str = Field(description="Place name exactly as written")
    occurrence_type: str = Field(description="Primary, Secondary, or Uncertain")

class ExtractionResult(BaseModel):
    occurrences: List[RawOccurrence]

class TaxonomicInfo(BaseModel):
    valid_name: str
    phylum: str = ""
    class_: str = ""
    order_: str = ""
    family_: str = ""
    authorship: str = ""
    year: str = ""
    taxonomic_status: str = "Verified"
    data_source: str = ""

class GeospatialInfo(BaseModel):
    decimalLatitude: Optional[float] = None
    decimalLongitude: Optional[float] = None
    geometry_geojson: str = ""
    geocodingSource: str = ""

class EnrichedOccurrence(BaseModel):
    raw: RawOccurrence
    taxonomy: TaxonomicInfo
    geospatial: GeospatialInfo

class ReviewResult(BaseModel):
    approved: bool
    review_notes: str
    occurrence: EnrichedOccurrence

class FinalBatchResult(BaseModel):
    approved_occurrences: List[EnrichedOccurrence]


# Agents Definitions

extractor_agent = Agent(
    model='test', # Will be overridden at runtime based on UI
    output_type=ExtractionResult,
    system_prompt=(
        "You are a biodiversity data extraction expert. "
        "Extract structured JSON for EVERY species-occurrence event found in the text. "
        "Adhere strictly to the extraction rules: One species per record, one locality per record."
    )
)

taxonomic_agent = Agent(
    model='test',
    output_type=TaxonomicInfo,
    system_prompt=(
        "You are a Taxonomic Agent. Verify the scientific name. "
        "Use the tools provided to resolve taxonomy via gnfinder/gnparser. "
        "If you encounter a context where GBIF or the tool is wrong (e.g., Caulerpa flagged as Animalia "
        "but the genus context implies Plantae/Chlorophyta), you must override the classification."
    )
)

@taxonomic_agent.tool
def verify_taxonomy(ctx: RunContext[None], verbatim_name: str) -> dict:
    api = TaxonomicAPI()
    return api.verify_name(verbatim_name)

geospatial_agent = Agent(
    model='test',
    output_type=GeospatialInfo,
    system_prompt=(
        "You are a Geospatial Agent. You convert locality strings into coordinates and GeoJSON polygons. "
        "If the locality implies a region or linear feature (like 'Gulf of Kutch' or 'coastline'), "
        "use the tool to query for polygons (e.g., natural=coastline). You can specify a buffer radius."
    )
)

@geospatial_agent.tool
def geocode_locality(ctx: RunContext[None], locality_string: str, feature_hint: str = None, buffer_meters: int = None) -> dict:
    api = GeospatialAPI()
    return api.geocode(locality_string, feature_type=feature_hint, buffer_meters=buffer_meters)


reviewer_agent = Agent(
    model='test',
    output_type=ReviewResult,
    system_prompt=(
        "You are a Reviewer Agent. Evaluate the enriched occurrence data (raw + taxonomy + geospatial). "
        "Ensure there are no obvious discrepancies (e.g., a marine species located inland, or invalid taxonomy). "
        "Provide approval status and notes."
    )
)

expert_agent = Agent(
    model='test',
    output_type=FinalBatchResult,
    system_prompt=(
        "You are an Expert Marine Biologist. You make the final decision on a batch of reviewed occurrences. "
        "You filter out rejected ones, format the approved ones for DB insertion, and prepare them to be written to the Wiki."
    )
)

# Orchestration pipeline
async def run_multi_agent_pipeline(text: str, model_id: str, client=None) -> List[EnrichedOccurrence]:
    kw = {"model": model_id}

    # 1. Extraction
    ext_res = await extractor_agent.run(f"Extract occurrences from the following text:\n\n{text}", **kw)
    raw_occurrences = ext_res.data.occurrences

    enriched_batch = []

    # 2 & 3. Parallel Taxonomy and Geospatial
    for raw in raw_occurrences:
        import asyncio
        tax_task = taxonomic_agent.run(
            f"Verify this name: {raw.recorded_name}. Context: {raw.raw_text_evidence}", **kw
        )
        geo_task = geospatial_agent.run(
            f"Geocode this locality: {raw.verbatim_locality}. Habitat context: {raw.habitat}", **kw
        )

        tax_res, geo_res = await asyncio.gather(tax_task, geo_task)

        enriched = EnrichedOccurrence(
            raw=raw,
            taxonomy=tax_res.data,
            geospatial=geo_res.data
        )

        # 4. Review
        rev_res = await reviewer_agent.run(
            f"Review this enriched occurrence: {enriched.model_dump_json()}", **kw
        )

        if rev_res.data.approved:
            enriched_batch.append(rev_res.data.occurrence)

    # 5. Expert Final Decision
    if not enriched_batch:
        return []

    expert_res = await expert_agent.run(
        f"Final review of these approved occurrences: {[e.model_dump_json() for e in enriched_batch]}", **kw
    )

    return expert_res.data.approved_occurrences
