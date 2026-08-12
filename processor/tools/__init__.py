"""
processor.tools — shared, template-independent toolkits.

Each subpackage is a *tool family* grouped by the data contract it operates
on (not by clinical modality). Templates import a family as a library.

  ts_dsp/   voltage-timeseries DSP (EEG + iEEG): the Signal contract, the
            EDF reader, and the filtering / feature-extraction tools.

Future families (e.g. nifti_tools for imaging) live here too, each with the
same shape: a data contract + a tool registry + an apply_*_pipeline driver.
"""