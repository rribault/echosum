from setuptools import setup, find_packages

setup(
    name='echosum',
    version='0.0.1',
    description='record interview and summarize',
    author='Romain Ribault',
    author_email='rribault@gmail.com',
    license='GNU',
    install_requires=['sounddevice', 'soundfile', 'numpy', 'pytest', 'pydub', 'shiny', 'scipy', 'openai', 'watchdog', 'ffmpeg'],
    python_requires='>=3.11',
    packages=find_packages(where='src'),
    package_dir={'': 'src'}
)   