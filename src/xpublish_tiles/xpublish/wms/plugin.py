"""OGC Web Map Service XPublish Plugin"""

import json
from enum import Enum
from io import BytesIO
from typing import Annotated

import cf_xarray  # noqa: F401
import numpy as np
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from PIL import Image
from xpublish import Dependencies, Plugin, hookimpl

import xarray as xr
from xpublish_tiles.pipeline import pipeline
from xpublish_tiles.types import OutputBBox, OutputCRS, QueryParams
from xpublish_tiles.utils import lower_case_keys
from xpublish_tiles.xpublish.wms.types import (
    WMS_FILTERED_QUERY_PARAMS,
    WMSGetCapabilitiesQuery,
    WMSGetFeatureInfoQuery,
    WMSGetLegendGraphicQuery,
    WMSGetMapQuery,
    WMSQuery,
)
from xpublish_tiles.xpublish.wms.utils import create_capabilities_response


class WMSPlugin(Plugin):
    name: str = "wms"

    dataset_router_prefix: str = "/wms"
    dataset_router_tags: list[str | Enum] = ["wms"]

    @hookimpl
    def dataset_router(self, deps: Dependencies):
        """Add wms routes to the dataset router"""
        router = APIRouter(
            prefix=self.dataset_router_prefix, tags=self.dataset_router_tags
        )

        @router.get("", include_in_schema=False)
        @router.get("/")
        async def get_wms(
            request: Request,
            wms_query: Annotated[WMSQuery, Query()],
            dataset: xr.Dataset = Depends(deps.dataset),
        ):
            query_params = lower_case_keys(request.query_params)
            query_keys = list(query_params.keys())
            extra_query_params = {}
            for query_key in query_keys:
                if query_key not in WMS_FILTERED_QUERY_PARAMS:
                    extra_query_params[query_key] = query_params[query_key]
                    del query_params[query_key]

            match wms_query.root:
                case WMSGetCapabilitiesQuery():
                    return await handle_get_capabilities(request, wms_query.root, dataset)
                case WMSGetMapQuery():
                    return await handle_get_map(request, wms_query.root, dataset)
                case WMSGetFeatureInfoQuery():
                    return await handle_get_feature_info(request, wms_query.root, dataset)
                case WMSGetLegendGraphicQuery():
                    return await handle_get_legend_graphic(wms_query.root)

        return router


async def handle_get_capabilities(
    request: Request, query: WMSGetCapabilitiesQuery, dataset: xr.Dataset
) -> Response:
    """Handle WMS GetCapabilities requests with content negotiation."""

    # Determine response format from Accept header or format parameter
    accept_header = request.headers.get("accept", "")
    format_param = request.query_params.get("format", "").lower()

    # Default to XML for WMS compliance
    response_format = "xml"

    if format_param:
        if format_param in ["json", "application/json"]:
            response_format = "json"
        elif format_param in ["xml", "text/xml", "application/xml"]:
            response_format = "xml"
    elif "application/json" in accept_header:
        response_format = "json"

    # Get base URL from request
    base_url = str(request.url).split("?")[0]

    # Create capabilities response
    capabilities = create_capabilities_response(
        dataset=dataset,
        base_url=base_url,
        version=query.version,
        service_title="XPublish WMS Service",
        service_abstract="Web Map Service powered by XPublish and xarray",
    )

    if response_format == "json":
        # Return JSON response
        return Response(
            content=capabilities.model_dump_json(indent=2, exclude_none=True),
            media_type="application/json",
        )
    else:
        # Return XML response
        xml_content = capabilities.to_xml(
            xml_declaration=True, encoding="UTF-8", skip_empty=True
        )

        # Fix namespace prefixes for QGIS compatibility
        xml_str = (
            xml_content.decode("utf-8") if isinstance(xml_content, bytes) else xml_content
        )

        # Replace ns0: prefixes with default namespace for QGIS compatibility
        xml_str = xml_str.replace("ns0:", "")
        xml_str = xml_str.replace(
            'xmlns:ns0="http://www.opengis.net/wms"', 'xmlns="http://www.opengis.net/wms"'
        )

        # Ensure xlink namespace is present
        if "xmlns:xlink" not in xml_str and "xlink:" in xml_str:
            xml_str = xml_str.replace(
                'xmlns="http://www.opengis.net/wms"',
                'xmlns="http://www.opengis.net/wms" xmlns:xlink="http://www.w3.org/1999/xlink"',
            )

        xml_content = xml_str.encode("utf-8")

        return Response(
            content=xml_content,
            media_type="text/xml",
            headers={"Content-Type": "text/xml; charset=utf-8"},
        )


async def handle_get_map(
    request: Request, query: WMSGetMapQuery, dataset: xr.Dataset
) -> Response:
    """Handle WMS GetMap request."""

    # Extract dimension selectors from query parameters
    selectors = {}
    for param_name, param_value in request.query_params.items():
        # Skip the standard tile query parameters
        if param_name not in WMS_FILTERED_QUERY_PARAMS:
            # Check if this parameter corresponds to a dataset dimension
            if param_name in dataset.dims:
                selectors[param_name] = param_value

    # Special handling for time and vertical axes per wms spec
    if query.time or query.elevation:
        cf_axes = dataset.cf.axes
        if query.time:
            time_name = cf_axes.get("T", None)
            if len(time_name):
                selectors[time_name[0]] = query.time
        if query.elevation:
            vertical_name = cf_axes.get("Z", None)
            if vertical_name:
                selectors[vertical_name[0]] = query.elevation

    style = query.styles[0] if query.styles else "raster"
    variant = query.styles[1] if query.styles else "default"

    render_params = QueryParams(
        variables=[query.layers],  # TODO: Support multiple layers
        style=style,
        colorscalerange=query.colorscalerange,
        variant=variant,
        crs=OutputCRS(query.crs),
        bbox=OutputBBox(query.bbox),
        width=query.width,
        height=query.height,
        format=query.format,
        selectors=selectors,
        colormap=query.colormap,
    )
    buffer = await pipeline(dataset, render_params)

    return Response(buffer.getbuffer(), media_type="image/png")


async def handle_get_feature_info(
    request: Request, query: WMSGetFeatureInfoQuery, dataset: xr.Dataset
) -> Response:
    """Handle WMS GetFeatureInfo request."""
    from xpublish_tiles.grids import guess_grid_system
    from xpublish_tiles.lib import transformer_from_crs

    selectors = {}
    for param_name, param_value in request.query_params.items():
        if param_name not in WMS_FILTERED_QUERY_PARAMS:
            if param_name in dataset.dims:
                selectors[param_name] = param_value

    if query.time or query.elevation:
        cf_axes = dataset.cf.axes
        if query.time:
            time_name = cf_axes.get("T", None)
            if len(time_name):
                selectors[time_name[0]] = query.time
        if query.elevation:
            vertical_name = cf_axes.get("Z", None)
            if vertical_name:
                selectors[vertical_name[0]] = query.elevation

    x_pixel = query.x
    y_pixel = query.y

    pixel_width = (query.bbox.east - query.bbox.west) / query.width
    pixel_height = (query.bbox.north - query.bbox.south) / query.height

    x_coord = query.bbox.west + (x_pixel + 0.5) * pixel_width
    y_coord = query.bbox.north - (y_pixel + 0.5) * pixel_height

    try:
        var_name = query.query_layers
        if var_name not in dataset.data_vars:
            return Response(
                content=f'{{"error": "Variable {var_name} not found in dataset"}}',
                media_type="application/json",
                status_code=404,
            )

        da = dataset[var_name]

        if selectors:
            da = da.sel(selectors, method="nearest")

        grid = guess_grid_system(dataset, var_name)
        output_to_input = transformer_from_crs(crs_from=query.crs, crs_to=grid.crs)

        native_x, native_y = output_to_input.transform(x_coord, y_coord)

        cf_axes = da.cf.axes
        x_dim = cf_axes.get("X", [None])[0]
        y_dim = cf_axes.get("Y", [None])[0]

        x_coords = da[x_dim].values if x_dim else None
        y_coords = da[y_dim].values if y_dim else None

        debug_info = {
            "x_pixel": x_pixel,
            "y_pixel": y_pixel,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "native_x": native_x,
            "native_y": native_y,
            "x_dim": x_dim,
            "y_dim": y_dim,
            "grid_crs": str(grid.crs),
            "request_crs": str(query.crs),
            "x_coords_shape": str(x_coords.shape) if x_coords is not None else None,
            "y_coords_shape": str(y_coords.shape) if y_coords is not None else None,
            "grid_type": type(grid).__name__,
        }

        if x_dim and y_dim:
            selection_method = "unknown"
            selected_indices = {}
            try:
                point_value = da.sel({x_dim: native_x, y_dim: native_y}, method="nearest")
                selection_method = "sel_nearest"
            except (KeyError, ValueError) as e:
                x_coords = da[x_dim].values
                y_coords = da[y_dim].values

                if x_coords.ndim == 2 and y_coords.ndim == 2:
                    distances = (x_coords - native_x)**2 + (y_coords - native_y)**2
                    min_idx = distances.argmin()
                    idx_2d = divmod(min_idx, distances.shape[1])
                    selected_indices = {da.dims[0]: int(idx_2d[0]), da.dims[1]: int(idx_2d[1])}
                    point_value = da.isel(selected_indices)
                    selection_method = "curvilinear_2d"
                else:
                    x_idx = int(np.abs(x_coords - native_x).argmin())
                    y_idx = int(np.abs(y_coords - native_y).argmin())
                    selected_indices = {x_dim: x_idx, y_dim: y_idx}
                    point_value = da.isel(selected_indices)
                    selection_method = "1d_coords"

            value = float(point_value.values)
            debug_info["selection_method"] = selection_method
            debug_info["selected_indices"] = selected_indices

            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [x_coord, y_coord],
                        },
                        "properties": {
                            var_name: value,
                            "value": value,
                            "debug": debug_info,
                        },
                    }
                ],
            }

            return Response(
                content=json.dumps(geojson),
                media_type="application/json",
            )
        else:
            return Response(
                content='{"error": "Could not determine spatial dimensions"}',
                media_type="application/json",
                status_code=400,
            )

    except Exception as e:
        return Response(
            content=f'{{"error": "{str(e)}"}}',
            media_type="application/json",
            status_code=500,
        )


async def handle_get_legend_graphic(query: WMSGetLegendGraphicQuery) -> Response:
    """Handle WMS GetLegendGraphic request with a dummy PNG response."""

    # Create a simple dummy PNG image
    img = Image.new("RGB", (query.width, query.height), color="white")

    # Save to BytesIO buffer
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
    )
