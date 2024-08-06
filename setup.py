import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="pos_ner",
    version="1.0.0",
    author="Sayed Shaun",
    author_email="sayedshaun4@gmail.com",
    description="Part-Of-Speech and Named Entity Recognition System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SayedShaun/POS-NER.git",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'pandas',
        'matplotlib',
        'tqdm',
        'scikit-learn'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
