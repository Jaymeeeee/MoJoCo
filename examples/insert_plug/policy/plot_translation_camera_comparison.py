"""Plot exact saved third-person inputs, fixed-region zooms and RGB differences.

This scientific figure only reads the already captured scene arrays. It never
renders a new scene, runs a policy, or changes the original image content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DIRECTIONS = ('baseline', 'x_plus', 'x_minus', 'y_plus', 'y_minus')
LABELS = {'baseline': 'Original position', 'x_plus': '+X 30 mm', 'x_minus': '\N{MINUS SIGN}X 30 mm',
          'y_plus': '+Y 30 mm', 'y_minus': '\N{MINUS SIGN}Y 30 mm'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_peg_head(point, calibration):
    """Project saved peg-head site into integer-centred optical pixel coordinates."""
    world_from_camera = np.asarray(calibration['world_from_camera_optical_at_capture'])
    local = world_from_camera[:3, :3].T @ (point - world_from_camera[:3, 3])
    if local[2] <= 0:
        raise ValueError('Saved peg head is behind the optical camera')
    homogeneous = np.asarray(calibration['intrinsics']) @ local
    return homogeneous[:2] / homogeneous[2]


def pixel_summary(image, baseline, region):
    x0, y0, x1, y1 = region
    delta = np.abs(image[y0:y1, x0:x1].astype(float)-baseline[y0:y1, x0:x1].astype(float))
    return dict(rgb_mean_absolute_difference_0_255=float(delta.mean()),
                rgb_root_mean_square_difference_0_255=float(np.sqrt(np.mean(delta**2))),
                changed_pixel_fraction_at_5_levels=float(np.mean(np.max(delta, axis=2) >= 5)),
                pixels=int(delta.shape[0]*delta.shape[1]))


def create_figure(source_run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    scene_directory = source_run/'scenes'/'pos2'
    loaded, hashes = {}, {}
    bounds = []
    for direction in DIRECTIONS:
        npz, metadata_path = scene_directory/f'{direction}.npz', scene_directory/f'{direction}.json'
        hashes[str(npz)] = sha256(npz)
        hashes[str(metadata_path)] = sha256(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        with np.load(npz, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        image = arrays['observation.images.head']
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f'Expected exact uint8 256x256 head observation: {npz}')
        if metadata['calibration']['head']['physical_camera'] != 'third_person':
            raise ValueError('head is not the physical third-person camera')
        if metadata['observation_sha256'] != hashes[str(npz)]:
            raise ValueError('Saved observation does not match scene manifest hash')
        for object_name in ('source_holder_visual', 'peg_visual'):
            bounds.append(metadata['visibility']['objects'][object_name]['pixel_bounds'])
        uv = project_peg_head(arrays['peg'], metadata['calibration']['head'])
        loaded[direction] = dict(arrays=arrays, metadata=metadata, image=image, uv=uv)

    # One square crop encloses the projected screw+source meshes at all five
    # positions. The same world-camera pixels are used for every column; no
    # object-centred recropping is allowed because it would hide the translation.
    bounds = np.asarray(bounds)
    low = np.floor(bounds[:, :2].min(axis=0)).astype(int)-8
    high = np.ceil(bounds[:, 2:].max(axis=0)).astype(int)+9
    side = int(np.max(high-low))
    centre = (low+high)/2
    start = np.floor(centre-side/2).astype(int)
    start = np.clip(start, 0, 256-side)
    x0, y0 = start.tolist()
    x1, y1 = x0+side, y0+side
    roi = [x0, y0, x1, y1]
    output.mkdir(parents=True, exist_ok=True)
    raw_directory = output/'head_originals'
    raw_directory.mkdir(exist_ok=True)
    baseline = loaded['baseline']
    metrics = dict(
        source_run=str(source_run), physical_camera='third_person', policy_feature='observation.images.head',
        capture_time_s=0.0, source_images_shape=[256, 256, 3], source_image_dtype='uint8',
        raw_rgb_pngs_are_pixel_exact=True, no_new_rendering_or_model_calls=True,
        fixed_crop_xyxy_exclusive=roi,
        crop_definition='Union of projected screw and source-holder visual mesh bounds across all five positions, with margin; fixed square in image coordinates.',
        projection_definition='Projection of saved 3-D peg_head site through saved optical calibration; geometric location, not image-feature detection or segmentation.',
        changed_pixel_definition='At least one RGB channel differs by >=5 intensity levels on the 0..255 scale.',
        difference_image_definition='Mean absolute RGB channel difference from baseline, shown on one fixed 0..255 scale.',
        cautions=['Whole-frame image similarity is dominated by unchanged robot/table/background.',
                  'These are input-image differences, not evidence that the network can or cannot localize the object.',
                  'Both policies also receive wrist RGB and both cameras metric depth; this figure shows head RGB only.'],
        source_sha256=hashes, conditions={})
    fig, axes = plt.subplots(3, 5, figsize=(16.3, 10.5))
    fig.subplots_adjust(left=.055, right=.985, top=.885, bottom=.18, wspace=.07, hspace=.16)
    fig.suptitle('Third-person model inputs: original and four 30 mm source-assembly shifts',
                 x=.52, y=.98, fontsize=18, fontweight='bold')
    fig.text(.52, .941, 'Same robot, camera and target holder. Screw + source holder move together. Exact saved RGB inputs; no new rendering.',
             ha='center', fontsize=10.5, color='#354457')
    for column, direction in enumerate(DIRECTIONS):
        item = loaded[direction]
        image = item['image']
        np.testing.assert_array_equal(item['metadata']['calibration']['head']['intrinsics'],
                                      baseline['metadata']['calibration']['head']['intrinsics'])
        np.testing.assert_array_equal(item['metadata']['calibration']['head']['world_from_camera_optical_at_capture'],
                                      baseline['metadata']['calibration']['head']['world_from_camera_optical_at_capture'])
        path = raw_directory/f'{direction}.png'
        Image.fromarray(image).save(path)
        np.testing.assert_array_equal(np.asarray(Image.open(path)), image)
        uv_delta = item['uv']-baseline['uv']
        pixel_distance = float(np.linalg.norm(uv_delta))
        depth_diff = np.abs(item['arrays']['observation.depth.head'].astype(float)-
                            baseline['arrays']['observation.depth.head'].astype(float))*1000
        row = dict(source_xy_m=item['metadata']['source_xy'],
                   peg_head_pixel_uv=item['uv'].tolist(),
                   peg_head_pixel_delta_uv=uv_delta.tolist(),
                   peg_head_pixel_displacement=pixel_distance,
                   full_frame=pixel_summary(image, baseline['image'], [0, 0, 256, 256]),
                   fixed_crop=pixel_summary(image, baseline['image'], roi),
                   depth_full_frame_mean_absolute_difference_mm=float(depth_diff.mean()),
                   depth_fixed_crop_mean_absolute_difference_mm=float(depth_diff[y0:y1, x0:x1].mean()),
                   raw_png=str(path.relative_to(output)), raw_png_sha256=sha256(path))
        metrics['conditions'][direction] = row
        axes[0, column].imshow(image, interpolation='nearest')
        axes[0, column].set_title(LABELS[direction]+f'\nProjected screw-head shift: {pixel_distance:.1f} px', fontsize=10.5)
        axes[1, column].imshow(image[y0:y1, x0:x1], interpolation='nearest')
        difference = np.abs(image.astype(float)-baseline['image'].astype(float)).mean(axis=2)
        handle = axes[2, column].imshow(difference[y0:y1, x0:x1], cmap='magma', vmin=0, vmax=255, interpolation='nearest')
        axes[2, column].set_xlabel(f"RGB MAE: {row['fixed_crop']['rgb_mean_absolute_difference_0_255']:.1f} / 255\n"
                                  f"Changed crop pixels: {row['fixed_crop']['changed_pixel_fraction_at_5_levels']*100:.1f}%", fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0, 0].set_ylabel('Full image\n256 x 256', fontsize=11, labelpad=13)
    axes[1, 0].set_ylabel(f'Same fixed crop\n{side} x {side}, enlarged', fontsize=11, labelpad=13)
    axes[2, 0].set_ylabel('Absolute RGB difference\nfrom original, same crop', fontsize=11, labelpad=13)
    color_ax = fig.add_axes([.23, .105, .54, .014])
    colorbar = fig.colorbar(handle, cax=color_ax, orientation='horizontal')
    colorbar.set_label('Mean absolute channel difference (0..255); shared scale for every panel', fontsize=9)
    fig.text(.055, .03, f'Fixed crop: columns {x0}:{x1}, rows {y0}:{y1}. Nearest-neighbour display only. '
             'Changed pixel: any RGB channel differs by at least 5 / 255.', fontsize=9, color='#354457')
    fig.text(.055, .012, 'Global similarity can hide local movement; visual similarity alone does not diagnose policy reasoning. '
             'Both policies also receive wrist RGB and metric depth.', fontsize=9, color='#354457')
    paths = []
    for extension in ('png', 'pdf'):
        destination = output/f'third_person_translation_comparison.{extension}'
        fig.savefig(destination, dpi=200, bbox_inches='tight', facecolor='white')
        paths.append(str(destination))
    plt.close(fig)
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError('A source artifact changed while plotting')
    metrics['source_hashes_unchanged'] = True
    metrics['figures'] = paths
    (output/'pixel_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+'\n')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    source = args.source_run.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else source/'full64_initial'/'visuals'
    metrics = create_figure(source, output)
    print(json.dumps(dict(figures=metrics['figures'], conditions=metrics['conditions']), indent=2))


if __name__ == '__main__':
    main()
