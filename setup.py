from setuptools import setup

setup(
   name='src',
   version='1.0',
   description='A football prediction model',
   author='Alex',
   author_email='foomail@foo.example',
   packages=['src'],  #same as name
   install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)