from setuptools import setup

setup(
    name="generic-iterative-stemmer",
    version="0.1.9",
    description="A generic language stemming utility, dedicated for gensim word-embedding.",
    author="Asaf Kali",
    author_email="asaf.kali@mail.huji.ac.il",
    url="https://github.com/asaf-kali/generic-iterative-stemmer",
    install_requires=[
        "gensim~=4.1",
    ],
    extras_require={
        "training": [
            "numpy~=1.21",
            "pandas~=1.3",
            "matplotlib~=3.5",
            "fasttext~=0.9",
            "editdistance~=0.6",
            "tqdm~=4.62",
        ]
    },
    include_package_data=True,
    license="https://github.com/asaf-kali/generic-iterative-stemmer/blob/main/LICENSE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
