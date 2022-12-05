from setuptools import setup

install_requires = [

]

packages = [
    'TNPandMS_lib', 
    'TNPandMS_lib.optimizationProgram', 
    'TNPandMS_lib_cli',
]

console_scripts = [
    'TNPandMS_lib_cli=TNPandMS_lib_cli.call:main',
]

setup(
    name='TNPandMS_lib', 
    version='0.0.0', 
    packages=packages,
    install_requires=install_requires,
    entry_points={'console_scripts': console_scripts},
)