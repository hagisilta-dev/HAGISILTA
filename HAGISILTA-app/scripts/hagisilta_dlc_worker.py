# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import deeplabcut
import yaml

from hagisilta_runtime import detect_device
from hagisilta_labeled_video import create_lossless_labeled_videos


def main() -> int:
    parser = argparse.ArgumentParser(description="Background DeepLabCut actions for HAGISILTA")
    parser.add_argument(
        "action", choices=["extract", "extract-plan", "check", "merge", "analyze", "outliers", "labeled-video"]
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shuffle", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pcutoff", type=float, default=0.6)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()

    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    video = args.video.resolve() if args.video else None
    output = args.output.resolve() if args.output else None
    if video and not video.is_file():
        raise FileNotFoundError(video)
    if output:
        output.mkdir(parents=True, exist_ok=True)

    print(f"ACTION={args.action}", flush=True)
    print(f"CONFIG={config}", flush=True)
    if args.action == "extract":
        deeplabcut.extract_frames(
            str(config),
            mode="automatic",
            algo="kmeans",
            crop=False,
            userfeedback=False,
            opencv=True,
            videos_list=[str(video)] if video else None,
        )
    elif args.action == "extract-plan":
        if not args.plan or not args.plan.is_file():
            raise ValueError("extract-plan requires --plan")
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        base_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        for position, item in enumerate(plan.get("videos", []), start=1):
            planned_video = Path(item["path"]).resolve()
            frames = int(item["frames"])
            if not planned_video.is_file():
                raise FileNotFoundError(planned_video)
            temporary_config = dict(base_config)
            temporary_config["numframes2pick"] = frames
            temporary_handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".hagisilta-extract-",
                suffix=".yaml",
                dir=config.parent,
                delete=False,
            )
            temporary_path = Path(temporary_handle.name)
            try:
                temporary_handle.close()
                temporary_path.write_text(
                    yaml.safe_dump(temporary_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
                )
                print(
                    f"EXTRACT_VIDEO={position}/{len(plan['videos'])}:{planned_video.name}:{frames}", flush=True
                )
                deeplabcut.extract_frames(
                    str(temporary_path), mode="automatic", algo="kmeans", crop=False,
                    userfeedback=False, opencv=True, videos_list=[str(planned_video)],
                )
            finally:
                temporary_path.unlink(missing_ok=True)
    elif args.action == "check":
        deeplabcut.check_labels(str(config), draw_skeleton=True)
    elif args.action == "merge":
        deeplabcut.merge_datasets(str(config))
    elif args.action == "analyze":
        if not video or not output:
            raise ValueError("analyze requires --video and --output")
        deeplabcut.analyze_videos(
            str(config),
            [str(video)],
            shuffle=args.shuffle,
            trainingsetindex=0,
            save_as_csv=True,
            destfolder=str(output),
            batch_size=args.batch_size,
            device=detect_device(),
        )
    elif args.action == "labeled-video":
        if not video or not output:
            raise ValueError("labeled-video requires --video and --output")
        create_lossless_labeled_videos(
            str(config),
            [str(video)],
            shuffle=args.shuffle,
            trainingsetindex=0,
            destfolder=str(output),
            draw_skeleton=True,
            pcutoff=args.pcutoff,
            dotsize=4,
            overwrite=True,
        )
    elif args.action == "outliers":
        if not video:
            raise ValueError("outliers requires --video")
        deeplabcut.extract_outlier_frames(
            str(config),
            [str(video)],
            shuffle=args.shuffle,
            trainingsetindex=0,
            outlieralgorithm="uncertain",
            extractionalgorithm="kmeans",
            automatic=True,
            opencv=True,
            destfolder=str(output) if output else None,
        )
    print("ACTION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
