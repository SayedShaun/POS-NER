import setuptools

long_description = open("README.md", "r").read()
setuptools.setup(
    name="pos_ner",
    version="1.0.0",
    author="Sayed Shaun",
    author_email="sayedshaun4@gmail.com",
    description="Part-Of-Speech and Named Entity Recognition System",
    long_description=long_description,
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    requires=[
        'torch', 
        'pandas', 
        'matplotlib', 
        'tqdm', 
        'scikit-learn'
        ]
)