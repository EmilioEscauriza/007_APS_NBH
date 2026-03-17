# ============================================================
# Single entry point for workflow execution.
# Config: workflow_config.py
# Library: workflow_lib.py (merged raw_data_inspection + analysis + google_sheet_upload + correlation_analysis)
# ============================================================

import sys
from pathlib import Path

# Ensure this directory is on the path so aps_workflow_lib can import workflow_config, source_functions, etc.
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import workflow_config  # noqa: F401  # load config so lib sees it
import workflow_lib as lib

if __name__ == "__main__":

    # ---- raw_data_inspection ----
    # lib.data_structure_viewer()
    # lib.rocking_curve_data_structure_viewer()
    # lib.rocking_curve_rsm()
    # lib.raw_mask_oscillation_inspector()
    # lib.comparison_of_corr_and_g_ttc_plot_methods()
    # lib.compare_existing_vs_corr_entrypoint()
    # lib.compare_existing_ttc_and_cgpt_ttc_from_raw()
    # lib.compare_existing_ttc_and_ttc_from_raw()
    # lib.mask_roi_viewer_mp4_save()
    # lib.waterfall_roi_entrypoint()
    # lib.ttc_with_custom_mask()
    # lib.equal_q_map_ttc(n_rings=8)


    # ---- analysis (APS 08-IDE) ----
    # lib.h5_file_inspector(lib.h5_file)
    # lib.g2_plotter(lib.h5_file)
    # lib.ttc_plotter(lib.h5_file)
    # lib.intensity_vs_time(lib.h5_file)
    # lib.static_vs_dynamic_bins(lib.h5_file)
    lib.combined_plot(lib.h5_file)
    # lib.oauth_test()
    # lib.figure_upload()
    # lib.q_spacing_inspector(lib.h5_file)
    # lib.integrated_intensities_inspector(lib.h5_file)
    # lib.execute_find_bragg_peak_center_from_scattering_2d_with_overlay()
    # lib.exec_make_and_save_inferred_qphi_maps()
    # lib.exec_quick_check_inferred_qphi_npz()
    # lib.exec_bragg_peak_shape_metrics_fixed_q_phi()
    # lib.exec_build_q_phi_map()
    # lib.exec_integrated_intensities_plot()
    # lib.scroll_segments_ax0()

    # ---- google_sheet_upload ----
    # lib.exec_google_sheet_upload()
    # lib.exec_single_mask_plot_save()
    # lib.exec_mask_mesh_around_bright_peak()
    # lib.exec_g2_fitting()
    # lib.exec_q_dependent_ttc_plot()
    # lib.t_dep_xrd_argmax()

    # ---- correlation_analysis ----
    # lib.corr_cosine_fitting_test()
    # lib.corr_plot_of_lineout_directions()
    # lib.corr_plot_of_period_vs_diagonal_start()
    # lib.corr_plot_of_single_fft_antidiagonal_lineout()
    # lib.corr_plot_of_period_vs_diagonal_start_both_lineouts()
    # lib.corr_fft_2d_plot()
    # lib.corr_fft_2d_fitting_and_parameter_extraction()
    # lib.corr_plot_A4_17scan_central_brightest_ttcs()
    # lib.corr_exec_plot_3x5_brightest_plus_offsets_ttcs()
    # lib.corr_exec_plot_3x4_brightest_plus_offsets_ttcs()
    # lib.demo_with_random()
    # lib.spatial_scaling_calculator(center_mask=150, relative_mask=150, xray_energy_keV=12.4)
    # lib.spatial_scale_demo_plot(xray_energy_keV=12.4)
    # lib.normalisation_comparison()
    # lib.mask_partition_investigation(n_cols=4, n_rows=4)

    pass
