"""Spatial analysis tools for Chicago geospatial data.

These tools perform spatial operations on cached data to avoid
passing large GeoDataFrames through the MCP protocol.
"""

from typing import Dict, Any
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import json
import os
from pathlib import Path
from uuid import uuid4


from ddgs import DDGS

from server import mcp_tool


def _save_plot_to_file_store(buf: io.BytesIO, filename: str) -> Dict[str, Any]:
    """Save a PNG buffer to the agent file store and return a download record."""
    storage_root = Path(os.getenv("AGENT_FILE_STORAGE_ROOT", "./agent_chat_files"))
    outputs_dir = storage_root / "outputs"
    metadata_dir = storage_root / "metadata"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    file_id = f"file_{uuid4().hex[:12]}"
    target = outputs_dir / f"{file_id}__{filename}"
    target.write_bytes(buf.read())

    record = {
        "file_id": file_id,
        "filename": filename,
        "kind": "output",
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "download_url": f"/agent/files/{file_id}/download",
    }
    (metadata_dir / f"{file_id}.json").write_text(
        json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return record

from tools.cache import _dataframe_cache


# "violent"/"property" are CATEGORIES spanning several Chicago primary_type values; an
# exact-match filter on them returns zero rows (the silent all-zero-map bug). Map them to
# the right value sets so a category query produces a correct map.
_CRIME_CATEGORIES = {
    "VIOLENT": {"HOMICIDE", "BATTERY", "ASSAULT", "ROBBERY", "CRIMINAL SEXUAL ASSAULT",
                "CRIM SEXUAL ASSAULT", "SEX OFFENSE", "HUMAN TRAFFICKING", "KIDNAPPING"},
    "PROPERTY": {"THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "CRIMINAL DAMAGE",
                 "CRIMINAL TRESPASS", "ARSON"},
}
_CRIME_CATEGORIES["VIOLENT CRIME"] = _CRIME_CATEGORIES["VIOLENT CRIMES"] = _CRIME_CATEGORIES["VIOLENT"]
_CRIME_CATEGORIES["PROPERTY CRIME"] = _CRIME_CATEGORIES["PROPERTY CRIMES"] = _CRIME_CATEGORIES["PROPERTY"]


@mcp_tool(category="computation")
def count_crimes_per_community(crime_type: str = None) -> Dict[str, Any]:
    """
    Performs a spatial join to count crimes in each Chicago community area.
    
    Args:
        crime_type: Optional crime type to filter (e.g., "THEFT", "BATTERY"). 
                   If None, counts all crimes.
    
    Returns:
        dict: Summary with:
            - total_communities: Number of community areas
            - crime_counts: Dictionary mapping community name to crime count
            - top_communities: Top 10 communities by crime count
            - filtered_by: Crime type if filtered
    """
    from tools.data_tools import load_chicago_community_areas, load_chicago_crime_data

    if 'chicago_community_areas' not in _dataframe_cache:
        load_chicago_community_areas()
    
    if 'chicago_crime_data' not in _dataframe_cache:
        load_chicago_crime_data()
    
    gdf_polygons = _dataframe_cache['chicago_community_areas']
    gdf_points = _dataframe_cache['chicago_crime_data']
    
    # Filter by crime type if specified — accept a CATEGORY ("violent crime") or an exact
    # primary_type. A zero-match filter returns an explicit error instead of silently
    # producing an all-zero map.
    if crime_type and "primary_type" in gdf_points.columns:
        types_upper = gdf_points["primary_type"].astype(str).str.upper()
        key = str(crime_type).strip().upper()
        category = _CRIME_CATEGORIES.get(key)
        gdf_points = gdf_points[types_upper.isin(category)] if category is not None else gdf_points[types_upper == key]
        if gdf_points.empty:
            avail = sorted(set(_dataframe_cache['chicago_crime_data']["primary_type"].astype(str).str.upper()))
            return {
                "error": f"crime_type '{crime_type}' matched 0 incidents — not mapping an empty result.",
                "hint": "Pass an exact primary_type, or a category (" + ", ".join(sorted(_CRIME_CATEGORIES)) + ").",
                "available_primary_types": avail[:40],
                "filtered_by": key,
                "total_crimes": 0,
            }
    
    # Perform spatial join
    gdf_points = gdf_points.to_crs(gdf_polygons.crs)
    joined_gdf = gpd.sjoin(gdf_polygons, gdf_points, how="left", predicate="contains")
    
    # Count only matched points; left-join placeholder rows have null index_right.
    point_counts = joined_gdf.groupby("community")["index_right"].count()
    point_counts.name = "crime_count"
    
    # Merge back to polygons
    result_gdf = (
        gdf_polygons.merge(
            point_counts, left_on="community", right_index=True, how="left"
        )
        .fillna(0)
        .copy()
    )
    result_gdf["crime_count"] = result_gdf["crime_count"].astype(int)
    
    # Cache result for plotting
    _dataframe_cache['crime_counts_by_community'] = result_gdf
    
    # Create summary
    crime_dict = dict(zip(result_gdf["community"], result_gdf["crime_count"]))
    sorted_communities = sorted(crime_dict.items(), key=lambda x: x[1], reverse=True)
    
    summary = {
        "total_communities": len(result_gdf),
        "total_crimes": int(result_gdf["crime_count"].sum()),
        "crime_counts": crime_dict,
        "top_communities": [
            {"name": name, "count": int(count)} 
            for name, count in sorted_communities[:10]
        ],
        "_note": "Results cached. Use generate_crime_map() to visualize.",
        "_cache_key": "crime_counts_by_community"
    }
    
    if crime_type:
        summary["filtered_by"] = crime_type.upper()
    
    return summary


@mcp_tool(category="generation")
def generate_crime_map(title: str = "Crime Counts by Community Area", crime_type: str = None) -> Dict[str, Any]:
    """
    Render a CHOROPLETH of Chicago crime counts by COMMUNITY AREA (shaded polygons), loading
    and joining the data internally. This is an AREAL choropleth, NOT a point-density "heat
    map": if the user asked for a heat map / hotspot / kernel-density map of incident
    locations, use a point-density tool (e.g. heatmap_image on the crime points) instead.
    Do NOT call load_chicago_community_areas, load_chicago_crime_data, or count_crimes_per_community first.

    Args:
        title: Title for the map.
        crime_type: Optional filter — an exact primary_type (e.g. "THEFT") or a category
            ("violent crime", "property crime"). If it matches no incidents an error is
            returned instead of an empty map.

    Returns:
        dict: status + file_id/download_url for the PNG (or an error if the filter matched nothing).
    """
    # Always (re)compute counts for the requested filter so the map matches crime_type and a
    # zero-match filter surfaces an error rather than rendering a misleading empty map.
    counts = count_crimes_per_community(crime_type)
    if isinstance(counts, dict) and counts.get("error"):
        return counts

    gdf = _dataframe_cache['crime_counts_by_community']
    if "crime_count" not in gdf.columns or int(gdf["crime_count"].sum()) == 0:
        return {"status": "error",
                "error": "No crimes matched the request — not rendering an empty map.",
                "filtered_by": (str(crime_type).upper() if crime_type else None)}

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    gdf.plot(
        column="crime_count",
        ax=ax,
        legend=True,
        legend_kwds={"label": "Number of Crimes", "orientation": "horizontal"},
        cmap="YlOrRd"
    )
    ax.set_title(title, fontdict={"fontsize": "16", "fontweight": "3"})
    ax.set_axis_off()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)

    record = _save_plot_to_file_store(buf, "chicago_crime_map.png")
    return {
        "status": "success",
        "filename": record["filename"],
        "file_id": record["file_id"],
        "download_url": record["download_url"],
    }


@mcp_tool(category="retrieval_external")
def web_search_geo_links(topic: str, resource_type: str) -> list:
    """Plain WEB SEARCH that returns page LINKS about a topic — no datasets, no files, no geometry.

    Despite the geospatial wording it reads nothing and downloads nothing: it is a DuckDuckGo
    query returning titles, urls and snippets. web_search/web_fetch cover the open web more
    capably, opengeodata_search finds actual downloadable datasets, and overpass_search returns
    real map features with geometry.

    Args:
        topic: The topic to web-search.
        resource_type: 'datasets', 'notebooks' or 'publications' (only shapes the query).

    Returns:
        A list of web results (title, link, snippet) — links only, not data.
    """
    query_map = {
        "datasets": f"geospatial open data {topic}",
        "notebooks": f"jupyter notebook {topic} github",
        "publications": f"research paper {topic} pdf",
    }
    query = query_map.get(resource_type, f"geospatial {topic}")

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return results


@mcp_tool(category="computation")
def analyze_and_organize_results(results: list, topic: str) -> str:
    """
    Analyzes a list of search results and organizes metadata into a Markdown table.

    Args:
        results: A list of search results from the web_search_geo_links tool.
        topic: The original geospatial topic for context.

    Returns:
        A Markdown formatted string with the organized metadata.
    """
    if not results:
        return "No results to analyze."

    lines = ["| Title | Summary | URL |", "| --- | --- | --- |"]
    for item in results:
        title = str(item.get("title", "")).replace("|", "\\|")
        summary = str(item.get("body", "")).replace("|", "\\|")
        url = str(item.get("href", "")).replace("|", "\\|")
        if topic and topic.lower() not in (title + summary).lower():
            summary = f"{summary}"
        lines.append(f"| {title} | {summary} | {url} |")

    return "\n".join(lines)
