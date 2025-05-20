from setuptools import setup, find_packages

setup(
    name='housing_price_prediction',
    version='0.1.0',
    packages=find_packages(include=['housing', 'housing.*']),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'housing-viz=housing.visualization.runner:main',
            'housing-train=housing.modeling:main'
        ]
    },
    author='Your Name',
    description='Package for housing price prediction and visualizations'
)
