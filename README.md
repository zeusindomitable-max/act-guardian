<div align="center">

# **act-guardian**  
### **Prevent Dead Neurons in LLMs**

> **Aktivasi < 0.001 → suntik noise → hidupkan kembali.**

![PyPI](https://img.shields.io/pypi/v/act-guardian?color=green)
![Python](https://img.shields.io/badge/python-≥3.8-blue)

</div>

---

**Author**: **Hari Tedjamantri**  
**Email**: haryganteng06@gmail.com  
**X**: [@haritedjamantri](https://x.com/haritedjamantri)

---

## **Install**

```bash
pip install act-guardian
```
## Quick Start

```bash
from act_guardian import act_guardian

guard = act_guardian(model)  # min_act=1e-3, noise=1e-4

loss.backward()
guard()  # revive dead neurons
optimizer.step()
```
## Works Best With EBC

```bash
from ebc_clip import energy_budget_clip
clip = energy_budget_clip(model)
guard = act_guardian(model)

loss.backward()
clip()   # jaga gradien
guard()  # jaga aktivasi
optimizer.step()
```
## Results (LLaMA-7B)

Metric       | Without | With Act-Guardian
-------------|---------|-------------------
Dead Heads   | 3/32    | **0/32**
Final Loss   | 2.41    | **2.25**


<div align="center">
Made with  by **Hari Tedjamantri**
</div>
```


