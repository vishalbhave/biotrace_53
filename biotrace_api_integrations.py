import requests
import json
import sqlite3
import logging

logger = logging.getLogger("biotrace_api")

class TaxonomicAPI:
    def __init__(self, db_path="biodiversity_data/metadata_v5.db"):
        self.db_path = db_path

    def _get_from_cache(self, name):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM taxonomic_cache WHERE verbatim_name = ?", (name,)).fetchone()
            if row:
                return dict(row)
        return None

    def _save_to_cache(self, data):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO taxonomic_cache
                (verbatim_name, valid_name, phylum, class_, order_, family_, authorship, year, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("verbatim_name"), data.get("valid_name"), data.get("phylum"),
                data.get("class_"), data.get("order_"), data.get("family_"),
                data.get("authorship"), data.get("year"), data.get("data_source")
            ))

    def verify_name(self, verbatim_name: str) -> dict:
        cached = self._get_from_cache(verbatim_name)
        if cached:
            return cached

        # 1. Use gnparser to standardize and get authorship/year
        parsed_name = verbatim_name
        authorship = ""
        year = ""

        try:
            gnparser_url = "https://parser.globalnames.org/api/v1"
            resp = requests.post(gnparser_url, json=[verbatim_name], timeout=5)
            if resp.status_code == 200:
                res = resp.json()
                if res and len(res) > 0 and res[0].get("parsed"):
                    parsed_name = res[0]["canonical"]["full"]
                    authorship = res[0].get("authorship", {}).get("normalized", "")
        except Exception as e:
            logger.warning(f"gnparser API failed for {verbatim_name}: {e}")

        # 2. Use gnfinder (via GlobalNames verification API) to get higher taxonomy
        phylum = ""
        class_ = ""
        order_ = ""
        family_ = ""
        valid_name = parsed_name
        data_source = ""

        try:
            gnv_url = f"https://verifier.globalnames.org/api/v1/verifications/{parsed_name}"
            resp = requests.get(gnv_url, timeout=5)
            if resp.status_code == 200:
                res = resp.json()
                if res and "names" in res and res["names"]:
                    match = res["names"][0]
                    if match.get("bestResult"):
                        best = match["bestResult"]
                        valid_name = best.get("matchedName", parsed_name)
                        data_source = best.get("dataSourceTitleShort", "GNV")

                        if "classificationPath" in best and "classificationRanks" in best:
                            path = best["classificationPath"].split("|")
                            ranks = best["classificationRanks"].split("|")
                            for r, p in zip(ranks, path):
                                r_lower = r.lower()
                                if r_lower == "phylum": phylum = p
                                elif r_lower == "class": class_ = p
                                elif r_lower == "order": order_ = p
                                elif r_lower == "family": family_ = p
        except Exception as e:
            logger.warning(f"GNV API failed for {parsed_name}: {e}")

        result = {
            "verbatim_name": verbatim_name,
            "valid_name": valid_name,
            "phylum": phylum,
            "class_": class_,
            "order_": order_,
            "family_": family_,
            "authorship": authorship,
            "year": year,
            "data_source": data_source
        }

        self._save_to_cache(result)
        return result


class GeospatialAPI:
    def __init__(self, db_path="biodiversity_data/metadata_v5.db"):
        self.db_path = db_path

    def _get_from_cache(self, locality):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM locality_cache WHERE locality_string = ?", (locality,)).fetchone()
            if row:
                return dict(row)
        return None

    def _save_to_cache(self, locality_string, lat, lon, geojson, source):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO locality_cache
                (locality_string, decimalLatitude, decimalLongitude, geometry_geojson, source)
                VALUES (?, ?, ?, ?, ?)
            """, (locality_string, lat, lon, geojson, source))

    def geocode(self, locality_string: str, feature_type: str = None, buffer_meters: int = None) -> dict:
        cached = self._get_from_cache(locality_string)
        if cached:
            return cached

        lat = None
        lon = None
        geojson = ""
        source = ""

        if feature_type:
            try:
                from OSMPythonTools.overpass import Overpass
                overpass = Overpass()

                buffer_clause = f"(around:{buffer_meters})" if buffer_meters else ""

                query = f"""
                [out:json][timeout:25];
                area[name~"{locality_string}",i]->.searchArea;
                (
                  way[{feature_type}]{buffer_clause}(area.searchArea);
                  relation[{feature_type}]{buffer_clause}(area.searchArea);
                );
                out geom;
                """
                result = overpass.query(query)
                if result.elements():
                    el = result.elements()[0]
                    if el.geometry():
                        source = "Overpass"
                        coords = el.geometry()
                        if isinstance(coords, dict) and "coordinates" in coords:
                            geojson = json.dumps(coords)
                            if coords["type"] == "Point":
                                lon, lat = coords["coordinates"]
                            elif coords["type"] == "LineString":
                                lon, lat = coords["coordinates"][0]
                            elif coords["type"] == "Polygon":
                                lon, lat = coords["coordinates"][0][0]
            except Exception as e:
                logger.warning(f"Overpass failed for {locality_string}: {e}")

        if not geojson:
            try:
                url = f"https://nominatim.openstreetmap.org/search?q={locality_string}&format=jsonv2&polygon_geojson=1"
                headers = {'User-Agent': 'BioTrace/v5.4'}
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        best = data[0]
                        lat = float(best.get("lat"))
                        lon = float(best.get("lon"))
                        if "geojson" in best:
                            geojson = json.dumps(best["geojson"])
                        source = "Nominatim"
            except Exception as e:
                logger.warning(f"Nominatim failed for {locality_string}: {e}")

        result = {
            "locality_string": locality_string,
            "decimalLatitude": lat,
            "decimalLongitude": lon,
            "geometry_geojson": geojson,
            "source": source
        }

        if lat is not None or geojson:
            self._save_to_cache(locality_string, lat, lon, geojson, source)

        return result
