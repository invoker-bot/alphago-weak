#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_requirements():
    requirements = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


setup(
    name="alphago-weak",
    version="1.0.0",
    description="An experimental AlphaGo-like Go bot with PyTorch, MCTS, and GTP tools.",
    long_description=(ROOT / "readme.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "alphago=alphago_weak.cli:app",
        ],
    },
)
