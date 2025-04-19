from setuptools import setup, find_packages

setup(
    name="cyclonev2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "dash",
        "dash-bootstrap-components",
        "plotly",
        "sqlalchemy",
        "python-dotenv",
        "schedule",
        "lightgbm",
        "transformers",
        "praw",
        "tweepy",
        "pytz",
        "binance-connector",
        "requests",
        "tenacity"
    ]
)