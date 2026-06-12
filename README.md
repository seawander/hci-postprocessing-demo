# hci-postprocessing-demo
Demo codes in the post-processing of high contrast imaging datasets, GUI version.

For demonstration purposes only! 

1. RDI (reference differential imaging; see `./HST-code/rdi_gui.py`): target image - reference image 
2. ADI (angular differential imaging; see `./HST-code/adi_gui.py`): target image 1 - target image 2
3. PDI (polarization differential imaging; see `./SPHERE-PDI-code/pdi_demo.py`): pol direction 1 - pol direction 2, then local Stokes transform


Notes:
1. Some of the **concepts here are over-simplified** for teaching/demonstration purposes only.
2. run `python xxx.py` to run the codes, and you will see the buttons/sliders to click/change.

**DO NOT use the codes for actual research.**

What you will see after running the codes -- 

1. RDI demo below （in your terminal, run `python rdi_gui.py`):
<img width="1728" height="1022" alt="RDI" src="https://github.com/user-attachments/assets/725a218c-3eb4-4391-ad90-78465386abd4" />

2. ADI demo below（in your terminal, run `python adi_gui.py`):
<img width="1707" height="1002" alt="ADI" src="https://github.com/user-attachments/assets/30c9f967-23ed-4fc5-b773-3e654ac051de" />

3. PDI demo below（in your terminal, run `python pdi_demo.py`):
<img width="1728" height="1022" alt="PDI" src="https://github.com/user-attachments/assets/fc5ec104-7d2b-432d-b7a8-7226aa1ffd61" />



An ultimate goal could be joint (a) RDI, (b) PDI, and (c) ADI detection/characterization of a system, see the Beta Pic system below.
<img width="2400" height="819" alt="RDI-PDI-ADI" src="https://github.com/user-attachments/assets/7b689d6f-f597-40f6-930b-b808f20de669" />

The last figure is from: 本页最后一张图片来源：https://www.opticsjournal.net/Articles/OJ2312bfccbeaa1dea/Abstract

任彬, 杨凯宁, 董若冰, 孙赫. 天文学高对比度计算成像研究进展 [J]. _激光与光电子学进展_, 2026, 63(8): 0839002.

