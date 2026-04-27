# Unified Wiki Module
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime
import requests

logger = logging.getLogger("biotrace.wiki")

# ── Optional Dependencies ─────────────────────────────────────────────────────
_FOLIUM_AVAILABLE = False
try:
    import folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    pass

_SCIKIT_LLM_AVAILABLE = False
try:
    from skllm.models.gpt.text2text.format_tools import get_json_format
    _SCIKIT_LLM_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  SCIKIT-LLM SPECIES EXTRACTOR (Unified)
# ─────────────────────────────────────────────────────────────────────────────
class UnifiedSpeciesExtractor:
    """Extracts advanced morphological and spatial data using Scikit-LLM concepts."""
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def extract(self, text: str, occurrence: dict) -> dict:
        sp_name = occurrence.get("validName") or occurrence.get("recordedName", "")
        if not sp_name:
            return {}

        prompt = f"""
        Extract detailed biological facts about '{sp_name}' from the text below.
        Return ONLY a JSON object with these keys (use null or "" if missing):
        {{
            "type_locality": {{"verbatim": "", "lat": null, "lon": null}},
            "habitat_context": "",
            "depth_range_m": [],
            "coloration": {{"life": "", "preserved": ""}},
            "diagnostic_features": [],
            "size_metrics": {{"length_mm": null, "width_mm": null}},
            "voucher_specimens": [],
            "collector": "",
            "authority": "",
            "is_new_species": false
        }}

        TEXT:
        {text[:8000]}
        """
        try:
            raw = self.llm_fn(prompt)
            match = re.search(r"```*json?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
            return json.loads(raw)
        except Exception as exc:
            logger.debug(f"[WikiExtractor] Failed for {sp_name}: {exc}")
            return {}
        

# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED WIKI CLASS
# ─────────────────────────────────────────────────────────────────────────────
class UnifiedWiki:
    SECTIONS = ["species", "locality", "habitat", "taxonomy", "papers"]

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        for sec in self.SECTIONS:
            (self.root / sec).mkdir(parents=True, exist_ok=True)
        self._init_index()

    def _init_index(self):
        idx_path = self.root / "index.json"
        if not idx_path.exists():
            self._save_json(idx_path, {
                "total_articles": 0,
                "sections": {s: [] for s in self.SECTIONS},
                "last_updated": datetime.now().isoformat()
            })

    def _slugify(self, text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[\s_-]+", "-", text)

    def _read_json(self, path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_json(self, path: Path, data: dict):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Automatic Geocoding (OSM Overpass) ───────────────────────────────────
    def _overpass_geocode(self, locality: str) -> tuple[float|None, float|None]:
        """Fetch coordinates from OpenStreetMap via Overpass API."""
        if not locality or locality.lower() in ("not reported", "unknown"):
            return None, None
        
        if len(locality) > 100:  # Skip long morpho descriptions
            return None, None 

        logger.info(f"[OSM Overpass] Geocoding: {locality}")
        query = f"""
        [out:json][timeout:5];
        (
          node["name"~"(?i){locality}"];
          way["name"~"(?i){locality}"];
          relation["name"~"(?i){locality}"];
        );
        out center 1;
        """
        try:
            r = requests.get("https://overpass-api.de/api/interpreter", params={"data": query}, timeout=5)
            r.raise_for_status()
            data = r.json()
            if data.get("elements"):
                elem = data["elements"][0]
                lat = elem.get("lat") or elem.get("center", {}).get("lat")
                lon = elem.get("lon") or elem.get("center", {}).get("lon")
                if lat and lon:
                    return float(lat), float(lon)
        except Exception as e:
            logger.debug(f"[OSM] Overpass geocode failed for {locality}: {e}")
        return None, None


    # ── Ingestion ─────────────────────────────────────────────────────────────
    def update_from_occurrences(
        self, occurrences: list[dict],
        citation: str = "",
        llm_fn = None,
        update_narratives: bool = False,
        extra_facts_map: dict = None
    ) -> dict:
        counts = {"species": 0, "locality": 0}
        extra_facts_map = extra_facts_map or {}

        for occ in occurrences:
            sp_name = occ.get("validName") or occ.get("recordedName")
            if sp_name:
                self.update_species_article(
                    sp_name, occ, citation, 
                    extra_facts=extra_facts_map.get(sp_name, {})
                )
                counts["species"] += 1

            loc = occ.get("verbatimLocality")
            if loc and loc != "Not Reported":
                self.update_locality_article(loc, sp_name, occ, citation)
                counts["locality"] += 1

        self._rebuild_index()
        return counts

    def update_species_article(self, sp_name: str, occurrence: dict, citation: str, extra_facts: dict = None):
        slug = self._slugify(sp_name)
        fp = self.root / "species" / f"{slug}.json"
        art = self._read_json(fp)

        if not art:
            art = {
                "title": sp_name,
                "type": "species",
                "occurrences": [],
                "habitats": [],
                "depth_range_m": [],
                "provenance": [],
                "enhanced_facts": {},
                "created_at": datetime.now().isoformat()
            }

        # Deduplicate occurrences
        occ_hash = hash(f"{occurrence.get('verbatimLocality')}_{citation}")
        existing_hashes = [hash(f"{o.get('verbatimLocality')}_{o.get('Source Citation')}") for o in art["occurrences"]]
        
        if occ_hash not in existing_hashes:
            art["occurrences"].append(occurrence)

        if occurrence.get("habitat") and occurrence["habitat"] not in art["habitats"]:
            art["habitats"].append(occurrence["habitat"])

        # Merge enhanced facts
        if extra_facts:
            ef = art.get("enhanced_facts", {})
            for k, v in extra_facts.items():
                if v and not ef.get(k): 
                    ef[k] = v
                elif isinstance(v, list) and isinstance(ef.get(k), list):
                    ef[k] = list(set(ef[k] + v))
            art["enhanced_facts"] = ef

        prov_entry = {"citation": citation, "date": datetime.now().isoformat()}
        if prov_entry["citation"] not in [p["citation"] for p in art["provenance"]]:
            art["provenance"].append(prov_entry)

        art["last_updated"] = datetime.now().isoformat()
        self._save_json(fp, art)

    def update_locality_article(self, locality: str, sp_name: str, occurrence: dict, citation: str):
        slug = self._slugify(locality)
        fp = self.root / "locality" / f"{slug}.json"
        art = self._read_json(fp)

        if not art:
            art = {
                "title": locality,
                "type": "locality",
                "species_checklist": [],
                "decimalLatitude": occurrence.get("decimalLatitude"),
                "decimalLongitude": occurrence.get("decimalLongitude"),
                "created_at": datetime.now().isoformat()
            }
            # Auto-geocode if missing
            if art["decimalLatitude"] is None:
                lat, lon = self._overpass_geocode(locality)
                if lat:
                    art["decimalLatitude"] = lat
                    art["decimalLongitude"] = lon

        if sp_name and sp_name not in art["species_checklist"]:
            art["species_checklist"].append(sp_name)

        art["last_updated"] = datetime.now().isoformat()
        self._save_json(fp, art)

    def _rebuild_index(self):
        idx = {"total_articles": 0, "sections": {s: [] for s in self.SECTIONS}, "last_updated": datetime.now().isoformat()}
        for sec in self.SECTIONS:
            for fp in (self.root / sec).glob("*.json"):
                idx["sections"][sec].append(fp.stem)
                idx["total_articles"] += 1
        self._save_json(self.root / "index.json", idx)

    def get_species_article(self, sp_name: str) -> dict:
        return self._read_json(self.root / "species" / f"{self._slugify(sp_name)}.json")

    def index_stats(self) -> dict:
        return self._read_json(self.root / "index.json")

    def list_species(self) -> list[str]:
        return sorted(self._read_json(fp).get("title", "") for fp in (self.root / "species").glob("*.json"))

    def list_localities(self) -> list[str]:
        return sorted(self._read_json(fp).get("title", "") for fp in (self.root / "locality").glob("*.json"))

    # ── Folium Maps ───────────────────────────────────────────────────────────
    def generate_global_map(self):
        if not _FOLIUM_AVAILABLE: return None
        m = folium.Map(location=[20.0, 78.0], zoom_start=4, tiles="CartoDB positron")
        added = 0
        
        for fp in (self.root / "locality").glob("*.json"):
            art = self._read_json(fp)
            lat, lon = art.get("decimalLatitude"), art.get("decimalLongitude")
            
            if lat is not None and lon is not None:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    popup=f"<b>{art['title']}</b><br>{len(art.get('species_checklist',[]))} species",
                    color="#2E86AB", fill=True, fill_opacity=0.7
                ).add_to(m)
                added += 1
        return m if added > 0 else folium.Map(location=[20.0, 78.0], zoom_start=4)

    def generate_species_map(self, species_name: str):
        if not _FOLIUM_AVAILABLE: return None
        art = self.get_species_article(species_name)
        if not art: return None
        
        m = folium.Map(location=[20.0, 78.0], zoom_start=4, tiles="CartoDB positron")
        coords_added = []
        article_updated = False
        
        # 1. Plot Type Locality
        ef = art.get("enhanced_facts", {})
        if ef.get("type_locality", {}).get("lat"):
            lat = ef["type_locality"]["lat"]
            lon = ef["type_locality"]["lon"]
            coords_added.append((lat, lon))
            folium.Marker(
                [lat, lon],
                popup="Type Locality",
                icon=folium.Icon(color="red", icon="star")
            ).add_to(m)

        # 2. Plot Standard Occurrences
        for occ in art.get("occurrences", []):
            lat = occ.get("decimalLatitude")
            lon = occ.get("decimalLongitude")
            loc = occ.get("verbatimLocality", "")
            
            # Auto-Geocode via Overpass if missing
            if (lat is None or lon is None) and loc and loc != "Not Reported":
                lat, lon = self._overpass_geocode(loc)
                if lat and lon:
                    occ["decimalLatitude"] = lat
                    occ["decimalLongitude"] = lon
                    article_updated = True
            
            if lat is not None and lon is not None:
                coords_added.append((lat, lon))
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=loc,
                    color="#44BBA4", fill=True, fill_opacity=0.8
                ).add_to(m)
                
        # Save article if we found new coords
        if article_updated:
            self._save_json(self.root / "species" / f"{self._slugify(species_name)}.json", art)

        if not coords_added: return None
        
        # Fit bounds to points
        if len(coords_added) > 1:
            m.fit_bounds(m.get_bounds())
        elif len(coords_added) == 1:
            m.location = [coords_added[0][0], coords_added[0][1]]
            m.zoom_start = 8
            
        return m

    # ── UI Rendering ──────────────────────────────────────────────────────────
    def render_streamlit_infobox(self, species_name: str):
        """Native Streamlit Infobox render."""
        import streamlit as st
        art = self.get_species_article(species_name)
        ef = art.get("enhanced_facts", {})
        
        st.markdown(f"## 🐚 {art.get('title', species_name)}")
        if ef.get("authority"):
            st.caption(f"**Authority:** {ef['authority']}")
            
        c1, c2, c3 = st.columns(3)
        c1.metric("Recorded Habitats", len(art.get("habitats", [])))
        c2.metric("Known Localities", len(art.get("occurrences", [])))
        c3.metric("Type Specimen", "Yes" if ef.get("type_locality", {}).get("verbatim") else "No")

        if ef.get("diagnostic_features"):
            st.info("**Diagnostic Features:**\n" + "\n".join(f"• {f}" for f in ef["diagnostic_features"]))

    def render_species_markdown(self, species_name: str) -> str:
        art = self.get_species_article(species_name)
        if not art: return "*Article not found.*"

        ef = art.get("enhanced_facts", {})
        lines = [
            f"# {art.get('title', species_name)}",
            f"**Authority:** {ef.get('authority', 'Unknown')}",
            ""
        ]

        # Morphology Section
        if ef.get("coloration") or ef.get("size_metrics"):
            lines.extend(["## Morphology & Size", ""])
            col = ef.get("coloration", {})
            if col.get("life"): lines.append(f"- **Color in life:** {col['life']}")
            if col.get("preserved"): lines.append(f"- **Color preserved:** {col['preserved']}")
            
            size = ef.get("size_metrics", {})
            if size.get("length_mm"): lines.append(f"- **Length:** {size['length_mm']} mm")
            if size.get("width_mm"): lines.append(f"- **Width:** {size['width_mm']} mm")
            lines.append("")

        # Specimens Section
        if ef.get("voucher_specimens") or ef.get("collector") or ef.get("type_locality", {}).get("verbatim"):
            lines.extend(["## Specimen Records", ""])
            if ef.get("voucher_specimens"):
                lines.append(f"- **Vouchers:** {', '.join(ef['voucher_specimens'])}")
            if ef.get("collector"):
                lines.append(f"- **Collector:** {ef['collector']}")
            if ef.get("type_locality", {}).get("verbatim"):
                lines.append(f"- **Type Locality:** {ef['type_locality']['verbatim']}")
            lines.append("")

        # Localities Section
        lines.extend(["## Documented Occurrences", ""])
        for occ in art.get("occurrences", []):
            loc = occ.get("verbatimLocality", "Unknown")
            lat, lon = occ.get("decimalLatitude"), occ.get("decimalLongitude")
            coord_str = f" ({lat}, {lon})" if lat and lon else ""
            lines.append(f"- **{loc}**{coord_str} — {occ.get('habitat', 'Habitat unknown')}")

        lines.extend(["", "## Provenance"])
        for p in art.get("provenance", []):
            lines.append(f"- {p.get('citation')} ({p.get('date', '')[:10]})")

        return "\n".join(lines)