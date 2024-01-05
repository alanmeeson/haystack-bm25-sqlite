from setuptools import setup, find_packages

setup(
    name='Haystack BM25 SQLite',
    version='0.1.0',
    description='An SQLite backed BM25 KeywordDocumentStore for use with Haystack',
    author="Alan Meeson",
    author_email="alan@carefullycalculated.co.uk",
    install_requires=[
        'farm-haystack[sql]==1.*'
    ],
    packages=find_packages(
        where='.',
        include=['haystack_sqlite'],
        exclude=[]
    )
)
