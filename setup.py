from setuptools import setup

setup(name='twitter_sim',
      version='0.1',
      description='Simple Agent Based Model (ABM) for Twitter',
      author='David Beskow',
      author_email='dnbeskow@gmail.com',
      license='MIT',
      packages=['twitter_sim'],
      install_requires=[
              'netowrkx',
              'numpy',
              'scikit-learn',
              'progressbar2',
              'matplotlib',
              'pandas'
              ],
      # scripts=['bin/stream_content', 'bin/stream_geo'],
      include_package_data = True,
      zip_safe=False)
