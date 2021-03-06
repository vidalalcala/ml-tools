from codecs import open as codecs_open
from setuptools import setup, find_packages


# Get the long description from the relevant file
with codecs_open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(name='mltools',
      version='0.0.1',
      description=u'Machine Learning Tools',
      long_description=long_description,
      classifiers=[],
      keywords='',
      author=u'Vidal Alcala',
      author_email='vidal.alcala@gmail.com',
      url='https://github.com/vidalalcala/mltools',
      license='MIT',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'click',
          'rpy2==2.8.4',
          'scikit-learn',
          'scipy',
          'pandas',
          'numpy'
      ],
      extras_require={
          'test': ['pytest'],
      },
      entry_points="""
      [console_scripts]
      mltools=mltools.scripts.cli:cli
      """
      )
