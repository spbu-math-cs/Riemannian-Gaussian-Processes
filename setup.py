from setuptools import setup, find_packages

requirements = (
    'firedrake',      # not on PyPy
    'paramz>=0.9.5',
    'autograd>=1.3',
    'networkx>=2.4',
)

extra_requirements = {
}

setup(name='manifold_matern',
      version='0.1',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=requirements,
      extras_require=extra_requirements,)

