# Rank-based Modeling for Universal Packets Compression in Multi-modal Communications

### This repository contains partial code for the ByteTrans framework.


## ByteTrans Framework

<p align="center">
  <img src="images/ByteTrans.png" alt="ByteTrans Framework" width="400">
</p>



## An Example of Lossless Compression
The predictive model outputs a probability distribution over 256 possible byte values (0–255), with probabilities summing to 1. Each byte is assigned a unique rank based on its probability, where rank `0` corresponds to the most probable byte, rank `1` to the second most probable byte, and so on. This ensures every byte has a deterministic and unique rank.

Suppose the packet sequence needed to be compressed is `[120, 85, 90]`, the current input byte is `120`. The model outputs the following probabilities for the next byte:

| **Byte Value** | **Probability** | **Rank** |
|----------------|-----------------|----------|
| `85`           | `0.40`          | `0`      |
| `67`           | `0.30`          | `1`      |
| `90`           | `0.20`          | `2`      |
| ...            | ...             | ...      |

Here, the byte `85` is the most probable, so it is assigned rank `0`. During compression, only the rank (`0`) is stored instead of the actual byte value (`85`).

Now consider predicting the third byte, given the previous two bytes `[120, 85]`. The model outputs:

| **Byte Value** | **Probability** | **Rank** |
|----------------|-----------------|----------|
| `67`           | `0.35`          | `0`      |
| `90`           | `0.25`          | `1`      |
| `50`           | `0.15`          | `2`      |
| ...            | ...             | ...      |

The true value of the third byte is `90`, which corresponds to rank `1` based on the model’s predictions. I think this is what you called a “prediction error.” However, we do not store rank `0`. Instead, the rank `1` is stored, representing the byte `90`.

---

During decompression, the stored ranks are retrieved and mapped back to the original byte values using the same predictive model. Here’s how the process works for the given example:


Given the input byte `120` and the stored rank `0`, the predictive model outputs the following probabilities:

| **Byte Value** | **Probability** | **Rank** |
|----------------|-----------------|----------|
| `85`           | `0.40`          | `0`      |
| `67`           | `0.30`          | `1`      |
| `90`           | `0.20`          | `2`      |
| ...            | ...             | ...      |

Since the stored rank is `0`, the model retrieves the byte `85` as the most probable value.


Given the first two bytes `[120, 85]` and the stored rank `1`, the predictive model outputs the following probabilities:

| **Byte Value** | **Probability** | **Rank** |
|----------------|-----------------|----------|
| `67`           | `0.35`          | `0`      |
| `90`           | `0.25`          | `1`      |
| `50`           | `0.15`          | `2`      |
| ...            | ...             | ...      |

Since the stored rank is `1`, the model retrieves the byte `90`, which corresponds to the correct value.

---

By storing the ranks (`0` for `85` and `1` for `90`), the model ensures that the original packet sequence `[120, 85, 90]` is perfectly reconstructed. This deterministic mapping guarantees lossless compression, as the same predictive model is used during both compression and decompression to consistently interpret the ranks.


## Citation 

If you want to cite our paper in your work, please use the following BibTeX entry.

```bibtex
@inproceedings{xuanhaoluoByteTrans2025,
  title = {{Rank-Based Modeling for Universal Packets Compression in Multi-Modal Communications}},
  author = {Luo, Xuanhao and Peng, Zhiyuan and Li, Zhouyu and Yu, Ruozhou and Liu, Yuchen},
  booktitle = {{IEEE International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM)}},
  year = {2025},
}
