"""
CLM main source code.

This module contains the core Community Land Model (CLM) modules including
main control structures, decomposition modules, and primary model components.
"""

from .clm_driver import clm_drv, clm_drv_jit, CLMDriverState
from .decompMod import bounds_type
from .atm2lndType import atm2lnd_type, create_atm2lnd_instance, init_atm2lnd_arrays
from .clm_initializeMod import (
    initialize1, initialize2, full_initialize,
    initialize1_jit, initialize2_jit, full_initialize_jit,
    validate_initialization
)
from .abortutils import (
    endrun, handle_err, check_netcdf_status, assert_condition, 
    warn_and_continue, NetCDFConstants,
    CLMError, CLMNetCDFError, CLMInitializationError, CLMComputationError
)
from .clm_instMod import (
    clm_instInit, clm_instRest, clm_instInit_jit,
    atm2lnd_inst, soilstate_inst, waterstate_inst, canopystate_inst,
    temperature_inst, energyflux_inst, waterflux_inst, frictionvel_inst,
    surfalb_inst, solarabs_inst, mlcanopy_inst,
    CLMInstances, get_instance, reset_instances, validate_instances
)
from .clm_varcon import (
    # Mathematical constants
    rpi,
    # Physical constants
    tfrz, sb, grav, vkc, denh2o, denice, tkwat, tkice, tkair, 
    hfus, hvap, hsub, cpice, cpliq,
    # Bedrock constants
    thk_bedrock, csol_bedrock, zmin_bedrock,
    # Special values
    spval, ispval,
    # Classes
    PhysicalConstants, BedrockConstants, SpecialValues,
    # Utility functions
    get_jax_constants, get_constant, is_special_value, is_special_int_value,
    celsius_to_kelvin, kelvin_to_celsius, stefan_boltzmann_flux
)
from .clm_varctl import (
    iulog, CLMLogHandler, CLMRunControl, clm_run_control,
    set_log_unit, get_log_unit, log_message, close_log,
    TemporaryLogRedirect
)
from .clm_varpar import (
    # Module variables (Fortran compatibility)
    nlevsno, nlevsoi, nlevgrnd, numrad, ivis, inir, mxpft,
    # Classes and functions
    CLMParameters, clm_varpar_init, get_clm_parameters, set_custom_parameters,
    get_layer_info, get_radiation_info, get_pft_info, create_layer_arrays,
    reset_parameters, load_preset, CLM_PRESETS,
    # JAX utilities
    snow_layer_indices, soil_layer_indices, ground_layer_indices
)
from .ColumnType import (
    column_type, col, create_column_instance, reset_global_column,
    get_snow_layer_range, get_soil_layer_range, get_all_layer_range,
    get_active_snow_layers, get_soil_layer_mask, get_snow_layer_mask,
    create_layer_index_arrays
)
from .filterMod import (
    clumpfilter, filter, allocFilters, setFilters, setExposedvegpFilter,
    setExposedvegpFilter_jax, create_filter_instance, reset_global_filter,
    get_filter_indices, apply_patch_filter, apply_column_filter
)
from .GridcellType import (
    gridcell_type, grc, create_gridcell_instance, reset_global_gridcell,
    create_regular_grid, calculate_distance_haversine, degrees_to_radians,
    radians_to_degrees, normalize_longitude, validate_coordinates,
    create_coordinate_mesh
)
from .histFileMod import (
    hist_addfld1d, hist_addfld2d, HistoryManager, HistoryField,
    FieldType, AveragingFlag, ScaleType,
    get_history_manager, reset_history_manager
)
from .initGridCellsMod import (
    initGridcells, set_landunit_veg_compete, GridCellInitialization,
    get_grid_initialization_state, reset_grid_initialization,
    create_simple_grid, validate_patch_structure, calculate_grid_statistics,
    print_initialization_summary, initialize_single_patch_grid,
    initialize_multi_patch_grid, validate_initialization as validate_grid_init
)
from .initSubgridMod import (
    add_patch, SubgridStructure, get_subgrid_structure, reset_subgrid_structure,
    create_simple_subgrid, add_multiple_patches, validate_patch_hierarchy,
    get_patch_statistics, print_subgrid_summary, validate_subgrid_consistency,
    create_single_patch_subgrid, create_multi_column_subgrid
)
from .initVerticalMod import (
    initVertical, VerticalStructure, CLMPhysicsVersion,
    get_vertical_structure, reset_vertical_structure, calculate_layer_statistics,
    print_vertical_summary, validate_vertical_structure, create_simple_vertical_structure
)
from .ncdio_pio import (
    file_desc_t, ncd_pio_openfile, ncd_pio_closefile, ncd_inqdid, ncd_inqdlen,
    ncd_defvar, ncd_inqvdlen, ncd_io, ncd_io_1d, ncd_io_2d, ncd_double, ncd_int,
    NCDDataType, FileMode, NetCDFIOManager, get_netcdf_manager, reset_netcdf_manager,
    create_simple_netcdf_file, print_netcdf_summary
)
from .PatchType import (
    patch_type, patch, PFTType, VegetationCategory, PhotosynthesisType, PhenologyType,
    get_pft_name, get_vegetation_category, get_photosynthesis_type, get_phenology_type,
    is_vegetated, is_tree, is_shrub, is_grass, is_crop, is_irrigated, is_c3_plant, is_c4_plant,
    get_pft_statistics, create_patch_instance, reset_global_patch, validate_patch_structure,
    print_patch_summary, create_single_pft_patches, create_mixed_vegetation_patches
)
from .pftconMod import (
    pftcon_type, pftcon, get_photosynthesis_pathway, get_leaf_reflectance, get_vcmax,
    reset_pftcon, validate_pftcon, print_pftcon_summary, create_pftcon_subset
)

__all__ = [
    'clm_drv', 'clm_drv_jit', 'CLMDriverState', 'bounds_type',
    'atm2lnd_type', 'create_atm2lnd_instance', 'init_atm2lnd_arrays',
    'initialize1', 'initialize2', 'full_initialize',
    'initialize1_jit', 'initialize2_jit', 'full_initialize_jit',
    'validate_initialization',
    'endrun', 'handle_err', 'check_netcdf_status', 'assert_condition', 
    'warn_and_continue', 'NetCDFConstants',
    'CLMError', 'CLMNetCDFError', 'CLMInitializationError', 'CLMComputationError',
    'clm_instInit', 'clm_instRest', 'clm_instInit_jit',
    'atm2lnd_inst', 'soilstate_inst', 'waterstate_inst', 'canopystate_inst',
    'temperature_inst', 'energyflux_inst', 'waterflux_inst', 'frictionvel_inst',
    'surfalb_inst', 'solarabs_inst', 'mlcanopy_inst',
    'CLMInstances', 'get_instance', 'reset_instances', 'validate_instances',
    # Constants
    'rpi', 'tfrz', 'sb', 'grav', 'vkc', 'denh2o', 'denice', 'tkwat', 'tkice', 'tkair', 
    'hfus', 'hvap', 'hsub', 'cpice', 'cpliq', 'thk_bedrock', 'csol_bedrock', 'zmin_bedrock',
    'spval', 'ispval', 'PhysicalConstants', 'BedrockConstants', 'SpecialValues',
    'get_jax_constants', 'get_constant', 'is_special_value', 'is_special_int_value',
    'celsius_to_kelvin', 'kelvin_to_celsius', 'stefan_boltzmann_flux',
    # Control variables
    'iulog', 'CLMLogHandler', 'CLMRunControl', 'clm_run_control',
    'set_log_unit', 'get_log_unit', 'log_message', 'close_log',
    'TemporaryLogRedirect',
    # Parameters
    'nlevsno', 'nlevsoi', 'nlevgrnd', 'numrad', 'ivis', 'inir', 'mxpft',
    'CLMParameters', 'clm_varpar_init', 'get_clm_parameters', 'set_custom_parameters',
    'get_layer_info', 'get_radiation_info', 'get_pft_info', 'create_layer_arrays',
    'reset_parameters', 'load_preset', 'CLM_PRESETS',
    'snow_layer_indices', 'soil_layer_indices', 'ground_layer_indices',
    # Column types
    'column_type', 'col', 'create_column_instance', 'reset_global_column',
    'get_snow_layer_range', 'get_soil_layer_range', 'get_all_layer_range',
    'get_active_snow_layers', 'get_soil_layer_mask', 'get_snow_layer_mask',
    'create_layer_index_arrays',
    # Filters
    'clumpfilter', 'filter', 'allocFilters', 'setFilters', 'setExposedvegpFilter',
    'setExposedvegpFilter_jax', 'create_filter_instance', 'reset_global_filter',
    'get_filter_indices', 'apply_patch_filter', 'apply_column_filter',
    # Gridcell types
    'gridcell_type', 'grc', 'create_gridcell_instance', 'reset_global_gridcell',
    'create_regular_grid', 'calculate_distance_haversine', 'degrees_to_radians',
    'radians_to_degrees', 'normalize_longitude', 'validate_coordinates',
    'create_coordinate_mesh',
    # History file handling
    'hist_addfld1d', 'hist_addfld2d', 'HistoryManager', 'HistoryField',
    'FieldType', 'AveragingFlag', 'ScaleType',
    'get_history_manager', 'reset_history_manager',
    # Grid cell initialization
    'initGridcells', 'set_landunit_veg_compete', 'GridCellInitialization',
    'get_grid_initialization_state', 'reset_grid_initialization',
    'create_simple_grid', 'validate_patch_structure', 'calculate_grid_statistics',
    'print_initialization_summary', 'initialize_single_patch_grid',
    'initialize_multi_patch_grid', 'validate_grid_init',
    # Subgrid initialization
    'add_patch', 'SubgridStructure', 'get_subgrid_structure', 'reset_subgrid_structure',
    'create_simple_subgrid', 'add_multiple_patches', 'validate_patch_hierarchy',
    'get_patch_statistics', 'print_subgrid_summary', 'validate_subgrid_consistency',
    'create_single_patch_subgrid', 'create_multi_column_subgrid',
    # Vertical structure initialization
    'initVertical', 'VerticalStructure', 'CLMPhysicsVersion',
    'get_vertical_structure', 'reset_vertical_structure', 'calculate_layer_statistics',
    'print_vertical_summary', 'validate_vertical_structure', 'create_simple_vertical_structure',
    # NetCDF I/O
    'file_desc_t', 'ncd_pio_openfile', 'ncd_pio_closefile', 'ncd_inqdid', 'ncd_inqdlen',
    'ncd_defvar', 'ncd_inqvdlen', 'ncd_io', 'ncd_io_1d', 'ncd_io_2d', 'ncd_double', 'ncd_int',
    'NCDDataType', 'FileMode', 'NetCDFIOManager', 'get_netcdf_manager', 'reset_netcdf_manager',
    'create_simple_netcdf_file', 'print_netcdf_summary',
    # Patch types and PFT classification
    'patch_type', 'patch', 'PFTType', 'VegetationCategory', 'PhotosynthesisType', 'PhenologyType',
    'get_pft_name', 'get_vegetation_category', 'get_photosynthesis_type', 'get_phenology_type',
    'is_vegetated', 'is_tree', 'is_shrub', 'is_grass', 'is_crop', 'is_irrigated', 'is_c3_plant', 'is_c4_plant',
    'get_pft_statistics', 'create_patch_instance', 'reset_global_patch', 'validate_patch_structure',
    'print_patch_summary', 'create_single_pft_patches', 'create_mixed_vegetation_patches',
    # PFT constants and parameters
    'pftcon_type', 'pftcon', 'get_photosynthesis_pathway', 'get_leaf_reflectance', 'get_vcmax',
    'reset_pftcon', 'validate_pftcon', 'print_pftcon_summary', 'create_pftcon_subset'
]