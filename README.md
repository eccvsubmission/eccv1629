#  ECCV submission 1629: Learning ego-vehicle driving intentions from partially overlapping multi-modal datasets
The code and newly annotated NuScenes intention recognition labels accompanying our ECCV submission. 

<img src="figs/problem_overview.png" style="height: 400px; width:800px;"/>

## Start-up details
- pip install -r requirements.txt
- dataset pre-processing

## Datasets, Attribution & Licences 
**Todo: check appropriate license for sharing the annotations openly.**

Below an overview of the licenses and ways to get access to the datasets.

- Brain4Cars: http://brain4cars.com/

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution. (https://github.com/asheshjain399/ICCV2015_Brain4Cars/blob/master/LICENSE)

- HDD: https://usa.honda-ri.com/hdd (requires university affiliation and e-mail to request access to get the license)
- ROAD / Oxford: https://github.com/gurkirt/road-dataset

ROAD dataset is build upon Oxford Robot Car Dataset (OxRD). Similar to the original dataset (OxRD), the ROAD dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License and is intended for non-commercial academic use. If you are interested in using the dataset for commercial purposes, please contact original creator OxRD for video content and Fabio and Gurkirt for event annotations.
  
- NuScenes: https://www.nuscenes.org/ (also available at https://registry.opendata.aws/motional-nuscenes/)
  
Unless specifically labeled otherwise, these Datasets are provided to You under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (“CC BY-NC-SA 4.0”), with the additional terms included herein. The CC BY-NC-SA 4.0 may be accessed at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode When You download or use the Datasets from the Websites or elsewhere, You are agreeing to comply with the terms of CC BY-NC-SA 4.0 as applicable, and also agreeing to the Dataset Terms. Where these Dataset Terms conflict with the terms of CC BY-NC-SA 4.0, these Dataset Terms shall prevail. (https://www.nuscenes.org/terms-of-use)

### Model zoo
Todo: check simple way to host the trained weights, too large to upload here.

### Example usage
Check the run.py script for a full overview of possibilities.
NuScenes intention labels are available in the _data/labels_ folder.
You can easily wrap the run scripts in a bash file to execute a number of folds or combinations.

`python run.py --model-type [model type] --datasets hdd b4c oxford ...`
