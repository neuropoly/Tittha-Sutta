# Tittha-Sutta
Variety of blind metrics for MRI images.

## Installation
Option 1. Using `venv` and `pip`:
```
python -m venv .venv
source .venv/bin/activate
pip install .
```
Option 2. Using `uv`
```
uv venv
uv pip install .
```

After installation, `tittha-sutta` should be in your environment's PATH:
```
tittha-sutta -h
```

You can also run it with uv directly:
```
uv run tittha-sutta -h
```
## Usage
```
usage: tittha-sutta [-h] [-o OUT] nifti_dir
```

Example:
```
$ tittha-sutta data/niftis

image,slice,SNR_SC,SNR_WM,SNR_GM,CNR_WM_GM,ENTROPY
series-001.nii.gz,,,,,,
series-001.nii.gz,0,10.783260672102791,10.783260672102791,nan,nan,1694.2525638436036
series-001.nii.gz,1,7.727055795617609,7.727055795617609,nan,nan,1694.281762904902
series-001.nii.gz,2,8.41492042530151,8.41492042530151,nan,nan,1718.3552935372882
series-001.nii.gz,3,8.288534752839539,8.006520717950194,14.818008963809245,0.25683826447886915,1707.621135670652
series-001.nii.gz,4,7.686809253211473,7.686809253211473,nan,nan,1706.970973322263
series-001.nii.gz,5,8.851183392377822,8.879776981184198,12.59291909784119,0.5574645245782578,1704.3969176090652
series-001.nii.gz,6,6.848555326513129,6.848555326513129,nan,nan,1720.134914324897
series-001.nii.gz,7,8.56466601147805,8.56466601147805,nan,nan,1706.8582288668144
series-001.nii.gz,8,10.380995926810579,10.380995926810579,nan,nan,1707.9572604865716
series-001.nii.gz,9,9.136861318793489,9.336333372193126,17.2331027174926,0.7172494620573331,1701.0555801286073
series-001.nii.gz,10,10.649767714029611,10.683493584665126,19.424156791172923,0.6081124840167881,1713.4701903485632
series-001.nii.gz,11,12.753718114128343,13.058660897059932,15.444615273745672,0.49321132175432203,1712.5210727891608
series-001.nii.gz,12,9.554000823260125,9.587946145468628,13.602324292999718,0.4938937006474956,1706.9022851611007
series-001.nii.gz,13,7.625576322097904,8.435071917184805,13.991079960865694,1.0919029860215832,1733.4485312126244
series-001.nii.gz,14,11.165874519111853,11.474352432215035,20.035023744667118,0.7100589977739629,1705.7150431807468
```

## Add your own metrics
Create your custom python module with your own metric using the `@metric` decorator. The decorated function's name will be the name column's name in the CSV.
```python
### custom_metric.py ###
from tittha_sutta import metric, Context, ROI, tittha_sutta_main

import numpy as np

@metric(per_slice=True)
def CUSTOM_METRIC(ctx: Context) -> float:
    data = ctx.data # access slice data
    # access specific masked data
    spinalcord = ctx[ROI.spinalcord]
    graymatter = ctx[ROI.graymatter]
    whitematter = ctx[ROI.whitematter]

    # return float number
    return data.mean() / np.sqrt(spinalcord.std()**2 + graymatter.std()**2)

if __name__ == '__main__':
    tittha_sutta_main()
```
Run it:
```
$ python custom_metric.py nifti_dir
```
