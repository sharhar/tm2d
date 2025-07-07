from setuptools import setup, find_packages

setup(
    name='tm2d',
    version='0.1',
    packages=[
        "tm2d", 
        "tm2d.utilities",
        "tm2d.simulators",
    ],
    include_package_data=True,
    install_requires=[
        'Click',
        'numpy',
        'vkdispatch'
    ],
    entry_points='''
        [console_scripts]
        tm2d=tm2d.main:cli_entrypoint
    ''',
)
