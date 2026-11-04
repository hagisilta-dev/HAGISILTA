# HAGISILTA

English | [Korean](README.ko.md)

A DeepLabCut-based platform for behavioral experiment tracking and model development.

HAGISILTA stands for:

Human-Assisted Graphical Interface for
Systematic Inspection, Labeling, Tracking, and Analysis.

| H | A | G | I | S | I | L | T | A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human | Assisted | Graphical | Interface | Systematic | Inspection | Labeling | Tracking | Analysis |

The name describes a workflow that supports human review and correction throughout
labeling, tracking, and analysis.

## Main features

- Label video frames, train and evaluate pose models, and track bodyparts with DeepLabCut.
- Review predictions, correct coordinates, and compare postprocessing settings.
- Analyze Novel Object Recognition Test (NORT), Y-maze, Water maze, and general zone experiments.
- Define ROIs and analysis intervals, review multiple videos, and export portable projects.
- Use the application, editing windows, and installer in English or Korean.

## Distribution and runtime

HAGISILTA is a Windows desktop application. Its GUI, training, and inference
run in a dedicated Miniforge/Python environment. The application ZIP includes
the source, setup and launch tools, current assets, and translations.

## First installation on a new computer

1. Fully extract `HAGISILTA-app.zip`.
2. Check your internet connection and, if using an NVIDIA GPU, its graphics driver.
3. Double-click `HAGISILTA-Setup.cmd`.
4. HAGISILTA starts automatically when installation finishes.
5. For subsequent launches, use the `HAGISILTA` desktop shortcut.

Installation, repair, and management prompts follow the saved interface language.
If no language has been saved or supplied, the interactive installer asks you to
choose Korean or English. You can also run `HAGISILTA-Setup.cmd -Language en`
or `-Language ko`; the repair and management launchers accept the same option.
An explicit choice is saved once the Python environment is available and is
passed to environment checks and the application. Regular app launches never
show a language prompt. Third-party installation tools keep their own output language.

The installer automatically prepares a dedicated Miniforge installation, a
dedicated Python 3.12 environment, DeepLabCut 3.0.0, the HAGISILTA labeling tools,
PyTorch, and other required packages. If no NVIDIA GPU is detected, it installs
the CPU version, although training and inference may be very slow.

The dedicated environment is installed under
`%LOCALAPPDATA%\HAGISILTA\miniforge3`. The installer does not search for or modify
Python, Anaconda, Miniconda, or other Conda environments already installed on the
PC. The environment name is fixed as `hagisilta-dlc-3.0`, keeping it separate from
other Python environments used for research.

Installation requires an internet connection and at least 12 GB of free space on
the installation drive. At least 20 GB of free space is recommended for smooth
project work. A separate CUDA Toolkit installation is not required, but the
NVIDIA graphics driver must already be installed for GPU use.

## Launching and repairing the application

- `HAGISILTA-Run.cmd`: launch the installed HAGISILTA application.
- `HAGISILTA-Repair.cmd`: repair missing packages or update problems.
- `HAGISILTA-Menu.cmd`: open the launch, installation, repair, and environment-check menu.

Installation logs are saved in `%LOCALAPPDATA%\HAGISILTA\logs`.

For direct CLI use, run `hagisilta_ymaze_analysis.py single ...` or
`hagisilta_ymaze_analysis.py batch ...`. Use `hagisilta_pose_postprocess.py filter ...`
for pose postprocessing and `hagisilta_pose_postprocess.py apply ...` to apply
saved analysis inputs in a batch. Pass the command's arguments after its mode;
use `--help` after the mode to see the available arguments.

## Interface language

Choose Korean or English in the Language selector at the top of the main
window, then restart HAGISILTA to apply it throughout the application and its
editing and review windows. The choice is saved on this computer. Changing the
selector leaves the current session and running work open in their current language.
Translations work offline. Project names, filenames, user input, and analysis
data retain their saved values.

The English interface uses Fine-tuning for model training from existing weights
and Postprocessing for coordinate filtering. Unnamed-region defaults, QC warnings,
and saved review baseline labels use consistent English values. Review windows
show baseline descriptions in the selected interface language.

## Model workflow

1. Use `Check SuperAnimal models` on a representative video and its ROIs to
   assess bodypart detection under your recording conditions.
2. If the results are suitable, process the full videos in `Video tracking`.
   Otherwise, create labels in `Dataset configuration`, train a model, and
   assess it in `Model evaluation`.
3. Review the tracking results, correct coordinates or apply postprocessing as
   needed, then select the coordinates for behavior analysis.
4. Set up the experiment's regions and analysis intervals, and run the analysis.
   Rerun analysis after changing its inputs or settings to update the results.

## Video playback and review videos

Video correction, inference review, frame labeling, postprocessing comparison,
and the analysis interval and decision review windows for NORT, Y-maze, and
Water maze provide a `Playback speed` selector. The default is 1×; the available
speeds are 0.25×, 0.5×, 1×, 1.5×, 2×, and 4×. Use `Ctrl+[` to slow down and
`Ctrl+]` to speed up.

Speed changes apply immediately and are retained when switching videos or ROIs.
One-frame and one-second seeks remain the same at every speed. Playback speed
affects viewing only. Frames are not skipped, so actual playback may be slower
than the selected speed on demanding videos.

Review videos with bodyparts, connecting lines, and detection boxes are saved
as lossless FFV1 AVI files in Model evaluation, Video tracking, and SuperAnimal
model checks. They may be larger than MP4 files. Every saved frame is verified;
verification details are stored in `.lossless.json`. Odd dimensions receive at
most 1 px of padding on the right or bottom. Regenerating a review video creates
an AVI from the ROI video and coordinates while keeping existing MP4 files.
Lossless saving preserves the images being encoded; it does not repair loss
already present in the source. Behavior-decision videos use their own output format.

## ROI videos and tracking progress

ROI videos use lossless FFV1 encoding and retain the ROI's dimensions, FPS,
frames, and pixel values. In `Video tracking > Detailed run settings`, choose
`Concurrent ROI preparation` and `ROI output verification` separately.
A concurrency value of 4 allows up to 4 ROIs to be created or verified at once;
it is not the number of simultaneous GPU inference jobs.

| Verification option | What is checked | When to use it |
| --- | --- | --- |
| Quick, the default for batch tracking | Codec, dimensions, FPS, recorded frame count, and pixels from up to 9 frames including the beginning, middle, and end | Routine runs where sampled pixel checking is sufficient |
| Thorough | Every saved frame, as well as the video properties | When you need complete frame verification or suspect a saving error |

Both options produce the same lossless ROI video. Quick verification leaves
unsampled frames unchecked. Bodypart review videos always receive full
verification. A newly generated ROI replaces the existing one only after its
selected checks pass. Creation or verification errors preserve completed files;
forcefully ending a process may leave an incomplete temporary folder.

The result table's `Status` column and execution log show ROI preparation,
inference, coordinate saving, review-video creation, and completion. Hover over
a source video's status to see each ROI. `Fully complete` means that requested
post-inference work, including review videos, has finished. With review-video
creation disabled, a run finishes after coordinate processing. Queue mode creates
review videos after all inference is finished.

Completion is recorded in `tracking-status.json` in the ROI output folder.
`Coordinates available` identifies coordinate files without a verified completed
run. Interrupted, failed, and unprocessed ROIs remain incomplete.

Videos with the same filename stem can conflict in ROI and analysis output
folders. Give them distinct names, or use automatic numbering when copying them
into the project. HAGISILTA reports conflicts before linked-video registration
or batch processing and preserves results belonging to a different source.

## Training image size

Under `Model training > Detailed training settings > Training input region`,
choose one of these modes:

- Automatic full-ROI sizing uses all Train images and the rotation and scale
  augmentation range to calculate the input dimensions, adding black padding
  where needed.
- `Crop around labels` takes an area of the chosen width and height without
  shrinking the full ROI. Crop positions vary while retaining every labeled
  bodypart and a 32 px margin.

Crop width and height must each be a multiple of 32 and at least 32. If the
chosen area cannot contain the Train labels under the maximum rotation and
scale enlargement, training is blocked and the required dimensions are shown.
Unlabeled body regions may fall outside the crop: check `Preview input` and leave
sufficient space. A smaller input can reduce GPU memory use; reducing
`Pose batch size` can help when memory is still insufficient. The size check
includes labels used for memory replay.

`Preview input` shows the augmented full image and crop position on the left,
and the training input with label guides on the right. The preview uses a fixed
crop position; training varies it. Blue outlines and label points are guides,
not part of the training image. Source images and labels are preserved.

Neither sizing mode downsizes the image to fit the input dimensions; rotation
and scaling remain part of augmentation. Validation, Test, and video inference
use the original ROI with fixed-position padding to multiples of 32, rather
than the manually chosen training crop or a crop based on ground-truth labels.

### Padding for video evaluation

In Model evaluation's video testing and QC section, `Test-video input padding`
applies to the current run:

| Option | Effect |
| --- | --- |
| `Model default` | Use the model's saved inference settings |
| `Minimum padding` | Add centered black borders to reach the smallest dimensions that are multiples of 32 |
| `Match training` | Use the model's recorded full-ROI training dimensions |

`Match training` is unavailable when the model lacks recorded dimensions, the
ROI exceeds them, or the model used cropping or full-image downscaling. The
page explains the reason; `Model default` and `Minimum padding` remain available.
Each run saves separate ROI and result files, with its padding option shown in
the result table. This option affects video testing, not quantitative
Validation/Test evaluation or the saved model.

The training folder's `hagisilta-input-layout.json` records the Train list and
input settings. To assess cropping, train a new Shuffle with existing labels
and compare it on held-out videos at the original ROI size. Cropping does not
guarantee improved detection accuracy. Use HAGISILTA to reproduce its training
procedure; the saved weights and inference settings also work with DLC inference.

## Coordinate postprocessing and analysis inputs

Open `04 Compare coordinate postprocessing` in Model evaluation or the completed
results under `04 Review results` in Video tracking. Select a representative
result for comparison, or use `Select all` and Ctrl/Shift for batch processing.
The targets are the blue-selected result rows, separate from input-video checkboxes.

1. Compare settings. Open `Compare and Build Recipe` for one result. Compare
   unfiltered coordinates with up to 3 median windows, such as 3, 5, and 7 frames,
   in 4 synchronized views. Saved postprocessed files are not needed for a preview.
2. Configure and process. Choose `Same for all bodyparts` with one median window,
   or `Per-bodypart settings` with a saved recipe. Click `Postprocess selected
   results` or the button showing the selected count. The tracking-results
   context menu provides the same action.
3. Review and correct. Select an output in `Saved results`, then use
   `Compare Saved Results` or `Correct from Postprocessed Result`. Use
   `Import new corrected frames` to add saved manual corrections to training.
   Behavior-analysis inputs are assigned separately in Video tracking.

### Recipes and comparisons

A recipe assigns `Keep unfiltered` or an odd median window of 3–31 frames to each
bodypart, independently of its show/hide checkbox. The recipe editor displays
unfiltered coordinates, `Settings preview`, and up to 2 comparison windows.
Enter a name and choose `Use These Settings` to save the project-wide recipe JSON
and select it in step 2 of the workbench. It does not yet create postprocessed
H5/CSV files. Cancelling keeps the previous selection and asks before discarding
unused changes.

Recipes appear in both workbenches. Test a recipe in Model evaluation, then
apply it to selected tracking results. Each filter uses unfiltered coordinates;
a shared filter need not be run first. A frame window covers different amounts
of time at different video FPS. Results with different bodypart-name sets cannot
use the same recipe. The preview explains if a different bodypart set or a video
shorter than the chosen window prevents switching.

Comparison views synchronize playback, frame stepping, wheel zoom, and
middle-button panning. Choose another result in the left list; switching pauses
playback at its first frame. `Open side-by-side comparison` is also available in
the result table's context menu. In saved-result comparison, `Filters to compare`
selects the outputs to display; videos without filtered outputs show unfiltered
coordinates only. Comparisons use results from the same video and unfiltered
coordinates, using the latest result for each window unless a run is selected
explicitly. Saved-result comparison is for review and does not change processing
or analysis settings.

Each postprocessing run saves separate H5 and CSV files and comparison history
under the project's `.hagisilta/pose-filter-tests`. Original predictions, manual
corrections, and training labels are preserved. Saving a recipe, postprocessed
coordinates, or manual corrections does not automatically assign behavior-analysis
inputs. If processing fails or is stopped, completed outputs remain available.
An error is reported if saving fails. Rerun unfinished ROIs, then select the
completed results to assign their analysis inputs.

### Using coordinates for behavior analysis

In Video tracking, `Use N for behavior analysis` assigns the next NORT, Y-maze,
or Water maze analysis input for each selected ROI. Its saved output must match
the postprocessing method and recipe chosen in step 2. For one ROI, an explicitly
selected matching output takes priority; otherwise, the latest matching output
is used. A batch uses each ROI's latest matching output. Use `Use Unfiltered`
to return to the original prediction coordinates.

Assignment checks every input before applying the selection. Missing or damaged
files, duplicate ROIs, and multiple model results selected for one ROI block it.
If an assigned source video or coordinate file changes or becomes damaged,
reassign the input. The application asks for this instead of silently substituting
other coordinates. File sources are available in video-cell tooltips.

Run analysis again after assigning inputs to recalculate the results. Assignment
preserves existing analysis outputs, model-evaluation scores, manual corrections,
and training labels. New results record the coordinate file and postprocessing
recipe used.

## Bodyparts used for analysis

NORT, Y-maze, Water maze, and general zone analysis offer the bodypart names in
the coordinate results checked for analysis. When their bodypart sets differ,
only shared exact names are available. The blue editing row is separate from
the analysis checkboxes. Tooltips show the source, mismatches, and arenas without
coordinate results.

If coordinates are not available yet, the list comes from the selected tracking
model. Name capitalization and order are preserved. Refreshing the same model's
list retains checked points or a fully cleared selection. A damaged explicitly
selected analysis input must be checked before use.

Water maze analysis stops and identifies the file and missing names if a chosen
bodypart is absent from the coordinates. It does not calculate the center from
only the remaining points or treat names such as `body_center` and `bodycenter`
as interchangeable.

## Y-maze analysis intervals

Use `Y-maze > Arena configuration > Edit analysis interval` to set the start and
end for one ROI. Enter times as `minutes:seconds.milliseconds`, or use `S/F` to
set the current frame as the start/end. Confirming an entered time seeks to it.
To find a point 10 minutes after the start, select `From analysis start` under
`Reference + Time` and enter `10:00.000`. `View position` seeks without changing
the interval; `Set as end` also updates its end.

The left list shows the video's ROIs with their thumbnails and intervals.
Clicking a row switches ROI at the same playback time. Checkboxes select targets
for copying an interval, separately from the current blue row and the main
page's analysis checkboxes. Choose `Apply this interval to N checked ROIs` to
copy it to the selected drafts, then adjust individual ROIs as needed.

`Ctrl+S` saves all edited ROIs; `Save and close` also exits. Saving multiple ROIs
asks you to confirm their old and new intervals. Unchecking an ROI does not
remove an edited draft from the changes being saved. Cancelling asks before
discarding unsaved changes. Saved intervals take effect on the next analysis
run, while source videos, coordinates, and existing results are retained.

## ROI review and experiment displays

### Bulk ROI review

Choose a reference card, then use `Select all`, `Select search results`,
`Select videos without ROIs`, or `Clear all`. The reference is excluded from the
targets. Existing settings are kept unless you select `Overwrite existing ROIs`.
The Water maze pool and platform overview uses the same selection workflow,
grouping multiple trials from one arena into a card. Open a video card to edit
it and return to the overview when finished.

### Water maze overlays

In the subject interval registration window, use `ROI overlays` above the video
to show or hide the saved pool, platform, quadrants, and wall zone independently.
If an ROI is missing, damaged, or belongs to another video or frame size, the
window asks you to check its settings.

In the trajectory and heatmap editor, `Wall zone` displays the inner boundary
of the analysis wall-preference zone as a dashed line. Visibility, color, and
width apply to both the preview and saved image. If saved criteria are
unavailable, the editor shows a message. These display controls leave ROI
geometry, analysis intervals, and numerical results unchanged.

### Y-maze boundaries

Use `Show tracking ROI (T)`, `Show maze outline (H)`, and `Show zone fill (J)`
to control the reference display. Center must span all three arm entrances.
Center and Arm share aligned boundaries; changes made in the editor are written
to the ROI file when saved. Display toggles leave the coordinates unchanged.

## Exporting and importing portable projects

Choose separately whether to include videos and whether to include tracking
and analysis results. Included videos can be outside the project folder. The
export also includes ROI profiles and project-specific input settings. Wait for
inference and correction work to finish before exporting. A missing video
selected for inclusion is reported, and no new ZIP is created.

On import, the project adjusts video, ROI, correction-history, and analysis-input
paths to the new location. It keeps saved correction coordinates, provenance,
and temporal-context settings. Files with the same name in different folders
remain separate; other projects' global settings are not imported.

If videos are omitted, existing paths can work on the same PC. On another PC,
use the dashboard's `Path recovery` to link the source or ROI video folder before
editing ROIs or reviewing videos. Multiple files with the same name require
confirmation. To transfer the videos in a ZIP, export again with videos included.

## Project data

Projects are stored separately from the application under
`Documents\HAGISILTA-work`, using the Documents location set by Windows. The
application distribution excludes research videos, ROIs, labels, trained models,
inference results, and personal projects.

To open an existing NORT project, select its `config.yaml` or import its project
ZIP.

## License

HAGISILTA's original code, documentation, translations, and listed application
assets are released under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for
attribution and [LICENSE_SCOPE.json](LICENSE_SCOPE.json) for the file inventory.
The [license guide](LICENSE_GUIDE.md) covers asset credits, external dependencies,
and redistribution notices.

External dependencies retain their own licenses. SuperAnimal-TopViewMouse
weights have separate academic and non-commercial conditions for both
pretrained inference and the supported training workflow. Read
[PRETRAINED_MODELS.md](PRETRAINED_MODELS.md) before downloading or using them.
The `Model terms` button in the inference, training, evaluation, and tracking
headers opens these conditions in the selected interface language.
These weights are downloaded separately and are not covered by HAGISILTA's
Apache-2.0 license.

## Reporting research use

If you use HAGISILTA in your research, please mention it in Materials and Methods
with the version and [repository URL](https://github.com/hagisilta-dev/HAGISILTA).

Example: "Videos were analyzed using HAGISILTA (version X.Y.Z;
https://github.com/hagisilta-dev/HAGISILTA)."
Replace `X.Y.Z` with the version used in your research.

