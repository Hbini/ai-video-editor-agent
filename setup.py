from setuptools import setup, find_packages
setup(name='ai-video-editor-agent', version='0.1.0', author='Hernane Bini', description='AI-Powered Video Editor Agent', packages=find_packages(), python_requires='>=3.9', install_requires=['opencv-python>=4.8.0', 'torch>=2.1.0', 'fastapi>=0.104.0', 'pytest>=7.4.0'])
