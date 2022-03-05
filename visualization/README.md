# Installation Instructions

## Using a Python Virtual Environment

Install `virtualenvwrapper` as [instructed here](https://virtualenvwrapper.readthedocs.io/en/latest/install.html#basic-installation)
if you haven't done so already.  Then follow the instructions below:

```
# First, create a python virtualenv with python3
mkvirtualenv -p `which python3` vispy
# Install dependencies
pip install -r requirements.txt
# Run program
python ./3dhist.py
```
