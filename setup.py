#! /usr/bin/env python

from os import path

from setuptools import find_packages, setup

from seqscore import __version__


def setup_package() -> None:
    root = path.abspath(path.dirname(__file__))
    with open(path.join(root, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    setup(
        name="seqscore",
        version=__version__,
        packages=find_packages(include=("seqscore", "seqscore.*")),
        # Package type information
        package_data={"seqscore": ["py.typed"]},
        # 3.7 and up, but not Python 4
        python_requires="~=3.7",
        license="MIT",
        description="SeqScore: Scoring for named entity recognition and other sequence labeling tasks",
        long_description=long_description,
        install_requires=[
            "attrs>=19.2.0",
            "click",
            "tabulate",
        ],
        entry_points="""
            [console_scripts]
            seqscore=seqscore.scripts.seqscore:cli
        """,
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        url="https://github.com/bltlab/seqscore",
        long_description_content_type="text/markdown",
        author="Constantine Lignos",
        author_email="lignos@brandeis.edu",
    )


if __name__ == "__main__":
    setup_package()
