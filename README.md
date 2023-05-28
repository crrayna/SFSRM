# Single-frame super-resolution microscopy（SFSRM）
Example data and demo of SFSRM for single-frame super resolution of microscopy images.<br>
## Requirements
SFSRM is built with Python and pytorch. 
* Python >= 3.7
* PyTorch >= 1.7
## Install
1. Clone repo<br>
    ```
    git clone https://github.com/crrayna/SFSRM.git
    ```
2. Install dependent packages<br>
    ```
    pip install -r requirements.txt
    ```
## Quick inference
1. Download pretrianed models <br>

Download pretrained models at https://drive.google.com/drive/folders/1UnaDwrt1FNSAUT_OlosqvoV4jsxIxIhi?usp=sharing and put them in the pretrained_network folder

2. Inference <br>
```
python test.py -opt options/test/test_example_microtubule.yml
```
Results are in the `results` folder
## Acknowledgements
The codes are based on [ESRGAN](https://github.com/XPixelGroup/BasicSR) and [unetgan](https://github.com/boschresearch/unetgan). Please also follow their licenses. Thanks for their awesome works.
## References
[1] Wang, Xintao, et al. "Esrgan: Enhanced super-resolution generative adversarial networks." Proceedings of the European conference on computer vision (ECCV) workshops. 2018. <br>
[2] Schonfeld, Edgar, Bernt Schiele, and Anna Khoreva. "A u-net based discriminator for generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. <br>
## Contact
If you have any questions, please email `meshyao@ust.hk`
