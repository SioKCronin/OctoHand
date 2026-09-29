"""Roll out the scripted grasp-and-place sequence."""

from __future__ import annotations

import argparse
import zlib
import struct
from pathlib import Path

import numpy as np

from octohand.env import OctoHandEnv
from octohand.scripted import ScriptedPolicy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Watch OctoHand pick up an object and move it to the goal.")
    parser.add_argument("--task", choices=("block", "egg", "pen"), default="block")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tentacles", type=int, default=3)
    parser.add_argument("--frames", type=Path, default=None, help="Directory for PNG frames.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    render_mode = "rgb_array" if args.frames else None
    env = OctoHandEnv(task=args.task, n_tentacles=args.tentacles, render_mode=render_mode)
    policy = ScriptedPolicy()
    obs, info = env.reset(seed=args.seed)
    total = 0.0
    last_phase = None
    frames = 0
    if args.frames:
        args.frames.mkdir(parents=True, exist_ok=True)

    for step in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(policy.act(env))
        total += reward
        if args.verbose and (policy.phase != last_phase or step % 40 == 0):
            print(
                f"t={step:3d} {policy.phase:7s} err={info['pos_error']:.3f} "
                f"align={info['align']:.2f} contacts={info['contacts']} z={info['object_z']:.3f}"
            )
            last_phase = policy.phase
        if args.frames and step % 8 == 0:
            try:
                _write_png(args.frames / f"frame_{frames:04d}.png", env.render())
                frames += 1
            except Exception as exc:
                print(f"Rendering failed ({exc}). Continuing without frames.")
                args.frames = None
        if terminated or truncated:
            break

    env.close()
    status = "success" if info["success"] else "finished"
    print(
        f"{args.task} seed={args.seed} {status}  steps={step + 1}  "
        f"return={total:.2f}  pos_error={info['pos_error']:.3f}  align={info['align']:.2f}"
    )


def _write_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


if __name__ == "__main__":
    main()
