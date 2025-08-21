# dnndk-zedboard
This repo is part of the work presented at the [6th South-East Europe Design Automation, Computer Engineering, Computer Networks and Social Media Conference (SEEDA-CECNSM 2021), Preveza, Greece, September 24th-26th 2021](https://seeda2021.uowm.gr/) and published in [IEEE Xplore](https://ieeexplore.ieee.org/document/9566259).
 
# Paper: "Workflow on CNN utilization and inference in FPGA for embedded applications"
## workflow diagram of the project
<img src="https://github.com/ICscholar/dnndk-zedboard/blob/main/workflow_diagram.png" widht="600" height="450">
下面把这张流程图按“离线→编译→上板→运行”的顺序细细拆解。它描述的是把一个用 TensorFlow 训练好的 CNN（以 MNIST 为例）部署到 Xilinx ZedBoard 上、在 DPU（Deep-learning Processing Unit）上做推理的完整流水线。

## 1) 离线阶段：数据与模型
1. **MNIST dataset**
   * 仅作为示例数据集。
   * 图里有两条细箭头：
     * **train** → 送给 TensorFlow 训练模型；
     * **test** → 以后在板子上做推理验证（不是训练）。

2. **CNN Model & Optimizations (TensorFlow)**
   * 在主机上训练并做基本优化（如结构调整、BN 融合、Dropout 只在训练中等）。
   * 产出一个浮点模型（通常是 `.pb` / SavedModel）。

3. **Compression (DECENT)**
   * **DECENT** 是 Xilinx 的量化/剪枝工具链（老版本名称，核心是把浮点模型转成 **INT8**）。
   * 需要一小部分“校准数据”（可用 MNIST 的一部分），计算量化尺度，得到**定点化模型**，体积更小、上板速度更快。

## 2) 硬件准备：在 Vivado 做 DPU
4. **Hardware Design in Vivado**
   * 在 Vivado 里搭一个包含 **DPU** 的硬件设计（选择 DPU 的架构、时钟、AXI 互连等）。

5. **Modify DPU parameters**
   * 按需求调 DPU 参数：如阵列大小、核数、缓存配置等（常见型号如 B4096/B2304 之类）。
   * 这些参数将影响编译器映射与最终性能。

6. **Synthesis & BitStream**
   * 综合、实现并导出 **Bitstream** 与硬件描述文件。
   * 流程图中特别标注会得到一个 **`.hwh`** 文件：这是“硬件信息（Hardware Handoff）”，编译器用它了解 DPU 的具体结构与地址映射。

## 3) 模型编译：把定点网络映射到 DPU
7. **Compilation (DNNC)**
   * **DNNC**（DPU Neural Network Compiler）读取：

     * 上一步的 **`.hwh`**（知道硬件长什么样），
     * **DECENT 输出的定点模型**（知道网络算子与张量精度），
     * 可选的目标频率/批大小等参数。
   * 产出 **可在 DPU 上运行的二进制**，图里标注为 **`(.elf)`**（常见命名如 `dpu_<model>.elf`），以及模型元数据。

## 4) 应用与运行时：DNNDK 与上板
8. **DNNDK & Application directory**
   * **DNNDK** 提供 DPU 运行

图里的 **“(ftp)”** 指的是用 **FTP（File Transfer Protocol，文件传输协议）** 通过网络把文件从主机拷到 ZedBoard。
在这个流程中，它提示：把 **模型 `dpu_*.elf`** 和 **应用可执行文件** 等产物通过 FTP 传到板子上的指定目录。
补充：
* 实际操作可用命令行 `ftp <板子IP>` 或图形工具（如 FileZilla）。
* 由于 **FTP 明文传输**，更安全的替代是 **SCP/SFTP**（基于 SSH）。

按“做什么—装什么—怎么跑—命令含义”梳理成中文说明，并补充一些易错点。

# 项目概述

* **仓库名**：dnndk-zedboard
* **用途**：展示如何把 **TensorFlow** 训练的 **CNN（以 MNIST 为例）** 通过 **Xilinx DNNDK** 工具链量化、编译成 **DPU** 可执行内核，并在 **ZedBoard（Zynq-7000）** 上做推理。
* **对应论文**：*Workflow on CNN utilization and inference in FPGA for embedded applications*（SEEDA-CECNSM 2021 / IEEE Xplore）。
* **如果学术使用**：按 README 提供的 BibTeX 引用即可。

---

# 目录（README 中的大纲）

* Installation（安装）

  * ZedBoard SD 卡配置（板端环境）
  * Host 端配置与内核生成（PC 侧工具）
  * 在 ZedBoard 上编译应用并运行推理
* Citation（引用）

---

# 一、ZedBoard 端：SD 卡与启动

1. **硬件平台**：ZedBoard（Zynq-7000 SoC）。
2. **拨码/跳线**：把 JP7–JP11、JP6 调到 **SD 卡启动**（硬件手册有图示）。
3. **系统镜像**：下载 **ZedBoard 专用 DNNDK 镜像**（README 给了 Xilinx 链接）。
4. **写卡**：解压后用 **balena Etcher** 写入 ≥8GB SD 卡。
5. **连接串口**，上电启动，能进到 DNNDK 的 Linux 环境即可。

> 这一步的产物：一张能启动的 SD 卡（含 Linux、驱动、DPU 运行时等）。

---

# 二、主机（PC）端：工具与环境

* **系统**：Ubuntu 18.04 LTS。
* **DNNDK 版本**：3.1（UG1327 v1.6 文档）。安装命令：

  ```bash
  sudo ./install.sh   # 在 dnndk/host_x86 目录下
  ```

  会装好 **DECENT（量化）**、**DNNC（编译器）**、**DDump/DLet** 等工具。
* **深度学习环境**：建议用 **Anaconda** 建虚拟环境。仓库提供了 `decent_ecsa_lab.yml`：

  ```bash
  conda env create -f decent_ecsa_lab.yml
  ```
* **TensorFlow 版本**：1.12（README 给了 py2.7/py3.6 的 wheel）。

  > 说明：用 **CPU** 也能跑，但相较 **GTX1050Ti(4GB)** 的 GPU，**延迟大约慢 5 倍**。

---

# 三、模型准备与精度校验

项目已经附带：

* **冻结图**（`freeze/frozen_graph.pb`）——由浮点 TF 模型导出；
* **生成校准/测试图片**的脚本 `generate_images.py`。

1. **验证浮点模型精度**

   ```bash
   python eval_graph.py \
     --graph ./freeze/frozen_graph.pb \
     --input_node images_in \
     --output_node dense_1/BiasAdd
   ```

   期望：Top-1 ≈ **0.9902**，Top-5 ≈ **0.9999**。

2. **DECENT 量化（INT8）**

   ```bash
   decent_q quantize \
     --input_frozen_graph ./freeze/frozen_graph.pb \
     --input_nodes images_in \
     --output_nodes dense_1/BiasAdd \
     --input_shapes ?,28,28,1 \
     --input_fn graph_input_fn.calib_input \
     --output_dir _quant
   ```

   * `--input_fn`：提供**校准数据**的函数（脚本中定义）。
   * 输出会在 **`_quant/`** 下生成：

     * `quantize_eval_model.pb`（评估用）
     * `deploy_model.pb`（**部署用**）
       量化后再验证一次（评估图）：

   ```bash
   python eval_graph.py \
     --graph ./_quant/quantize_eval_model.pb \
     --input_node images_in \
     --output_node dense_1/BiasAdd
   ```

   期望：Top-1 ≈ **0.9904**，Top-5 ≈ **0.9999**（与浮点基本一致）。

---

# 四、DNNC 编译：生成 DPU 内核（模型 .elf）

把量化后的部署模型映射到 DPU：

```bash
dnnc-dpu1.4.0 \
  --parser=tensorflow \
  --frozen_pb=_quant/deploy_model.pb \
  --dcf=zedboard.dcf \
  --cpu_arch=arm32 \
  --output_dir=_deploy \
  --net_name=mnist \
  --save_kernel \
  --mode=normal
```

参数解释：

* `--parser=tensorflow`：解析 TF 模型；
* `--frozen_pb`：量化后的 **部署图**；
* `--dcf=zedboard.dcf`：**硬件描述**（与 DPU 配置匹配）。README 已提供；也可用 **DLet** 通过 Vivado 的 **`.hwh`** 生成；
* `--cpu_arch=arm32`：ZedBoard 的 ARM 架构；
* `--output_dir=_deploy`：输出目录；
* `--net_name=mnist`：网络名（关系到生成文件的命名）；
* `--save_kernel`：导出可加载的内核；
* `--mode=normal`：常规优化模式。

产物在 **`_deploy/`**，核心是 **模型内核 `*.elf`**（注意：这是**模型二进制**，和应用程序的可执行文件不是一回事）。

---

# 五、在板子上编译应用并运行

1. **拷文件到板子**
   README 提示用 **FTP** 把 `mnist_zedboard_inference` 目录传到板上（更安全也可用 **SCP/SFTP**）。
2. **安装/编译/运行**

   * **Step 1** 安装：

     ```bash
     ./install.sh
     ```
   * **Step 2** 在 `samples/mnist` 下用 **Makefile** 编译应用。
   * **Step 3** 运行生成的可执行文件（应用会加载上面的**模型 .elf**），进行推理并按论文方式收集结果。

---

# 关键文件/文件夹对照

* `freeze/`：冻结的浮点模型（`frozen_graph.pb`）。
* `_quant/`：量化产物（`deploy_model.pb`、`quantize_eval_model.pb`）。
* `_deploy/`：DNNC 编译产物（**模型内核 `dpu_*.elf`** 等）。
* `mnist_zedboard_inference/`：板端应用源码与示例。
* `zedboard.dcf`：DPU 硬件描述（可由 `.hwh` 用 DLet 生成）。
* `decent_ecsa_lab.yml`：conda 环境文件。
* `pkgs/`：DNNDK 提供的依赖包集合。

---

# 最小化“跑通”清单

1. 按 README 做好 **SD 卡镜像+跳线**，能进板载系统。
2. PC 端安装 **DNNDK 3.1**，用 conda 按 `decent_ecsa_lab.yml` 建环境。
3. 运行 `generate_images.py` 生成校准/测试数据；用 `eval_graph.py` 验证浮点精度。
4. `decent_q quantize` 得到 `_quant/deploy_model.pb` 并复验量化精度。
5. `dnnc-dpu1.4.0 ...` 编译出 `_deploy` 下的 **模型 .elf**。
6. 把 **模型 .elf** 和 **应用目录** 传到板上；`install.sh` → `make` → 运行，观察推理结果与性能。

---

# 常见坑 & 提示

* **版本匹配**：DNNDK 版本、`zedboard.dcf`/`.hwh` 对应的 **DPU 配置** 必须一致。
* **预处理一致性**：训练/量化/推理的输入归一化、尺寸、通道顺序一致，否则精度会偏。
* **模型 vs 应用的 ELF**：一个是**模型内核**（DNNC 产物），一个是**程序可执行文件**（Makefile 编译）；不要混淆。
* **传输方式**：FTP 明文不安全，实际可用 **SCP/SFTP**。
* **CPU 仅推理**：能跑，但明显慢于文中 GPU 环境。

# Citation
If you use this work in academic research, please, cite it using the following BibTeX:
```
@INPROCEEDINGS{9566259,
  author={Flamis, Georgios and Kalapothas, Stavros and Kitsos, Paris},
  booktitle={2021 6th South-East Europe Design Automation, Computer Engineering, Computer Networks and Social Media Conference (SEEDA-CECNSM)},
  title={Workflow on CNN utilization and inference in FPGA for embedded applications: 6th South-East Europe Design Automation, Computer Engineering, Computer Networks and Social Media Conference (SEEDA-CECNSM 2021)},
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/SEEDA-CECNSM53056.2021.9566259}
}
```

![](https://img.shields.io/github/last-commit/ECSAlab/dnndk-zedboard?style=plastic)

Table of Contents
=================

  * [Installation](#Installation)
  	* [ZedBoard SD card configuration for DNNDK](#zedboard-sd-card-configuration-for-dnndk)
	* [Host configuration for DNNDK setup and inference kernel generation](#host-configuration-for-dnndk-setup-and-inference-kernel-generation)
	* [Application build and run inference on the ZedBoard](#application-build-and-run-inference-on-the-zedboard)
  * [Citation](#citation)

# Installation
In the following steps we describe the process to setup the development enviroment.

## ZedBoard SD card configuration for DNNDK

Follow the steps below to setup the FPGA board:

1. Our work is based on ZedBoard which is a complete development kit for designers interested in exploring designs using the Xilinx Zynq®-7000 SoC (read more specs: http://zedboard.org/product/zedboard ).

<img src="https://i.imgur.com/TPmQRCm.jpg" widht="450" height="450">

2. Verify the ZedBoard boot (JP7-JP11) and MIO0 (JP6) jumpers are set to SD card mode (as described in the Hardware Users Guide https://www.zedboard.org/documentation/1521). Jumpers should be in the following position. 

![](https://i.imgur.com/VTboA8m.jpg)

3. Download the Xilinx DNNDK image file compatible for ZedBoard
https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zedboard-dnndk3.1-image-20190812.zip.
4. Extract the image from the zip file and burn it on an SD card (8GB or higher is needed) using etcher https://www.balena.io/etcher/.
5. Connect the serial console.
6. Connect power and switch on the board.

## Host configuration for DNNDK setup and inference kernel generation

A typical computer with a x86 CPU has been used, loaded with Ubuntu 18.04 LTS. The Deep Neural Network Development Kit (DNNDK) package version 3.1 has been downloaded from the official Xilinx website (prior registration required) https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx_dnndk_v3.1_190809.tar.gz. Chapter 1 of the DNNDK User Guide (UG1327 - v1.6) includes all the details required to setup the environment.

The exact version of the tools that was used for the GPU setup is shown in the table below.



| Ubuntu  18.04 LTS | GPU      | CUDA 10.0, cuDNN 7.4.1 | 2.7 | tensorflow_gpu-1.12.0-cp27-cp27mu-linux_x86_64.whl |
|-------------------|----------|------------------------|-----|----------------------------------------------------|
|                   |          |                        | 3.6 | tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl  |
|                   | CPU Only | None                   | 2.7 | tensorflow_gpu-1.12.0-cp27-cp27mu-linux_x86_64.whl |
|                   |          |                        | 3.6 | tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl  |


The CPU configuration is also possible but latency of five times longer can be met when compared with the GTX1050TI (4GB) GPU card that has been used. 

The packages can be found in folder "pkgs" under dnndk root folder. It is suggested to use Anaconda to create the environment where the selected package will be installed. Further information can be found in DNNDK User Guide (UG1327 - v1.6).
After downloading and unpacking the DNNDK package, execute the `sudo ./install.sh` command under the host_x86 folder to install the DECENT,DNNC, DDump and DLet tools on the host.

The conda environment used in this exercise is saved as yml file (decent_ecsa_lab.yml) and the environment can be created with the command  `conda env create -f decent_ecsa_lab.yml`.

The freeze model has been generated from the floating point model implementation in tensorflow and for convenience it is provided inside folder "freeze". It is suggested to visit the paper for details how to generate it from tensorflow or the tensorflow documentation.

It is also required to generate the calibration and the test images. The "generate_images.py" will do this work.

The accuracy of the freeze model can be validated with the following command:

```shell=
$ python eval_graph.py \ 
--graph ./freeze/frozen_graph.pb \
--input_node images_in \
--output_node dense_1/BiasAdd
```

The expected result is
 Top 1 accuracy with validation set: 0.9902
 Top 5 accuracy with validation set: 0.9999

The DECENT tool of DNNDK should be used next to generate the quantized model. The following command will enable this process:

```shell=
$ decent_q quantize \
--input_frozen_graph ./freeze/frozen_graph.pb \
--input_nodes images_in \
--output_nodes dense_1/BiasAdd \
--input_shapes ?,28,28,1 \
--input_fn graph_input_fn.calib_input \
--output_dir _quant
```

The process may take some time to finish. Once quantization is completed, summary will be displayed, like the one below:

```console=
INFO: Checking Float Graph...
INFO: Float Graph Check Done.
INFO: Calibrating for 100 iterations...
100% (100 of 100) |############| Elapsed Time: 0:00:13 Time:  0:00:13
INFO: Calibration Done.
INFO: Generating Deploy Model...
INFO: Deploy Model Generated.
*********** Quantization Summary **************      
INFO: Output:       
  quantize_eval_model:      quantize_eval_model.pb
  deploy_model:                    deploy_model.pb
```  

At this point, it is suggested to validate the accuracy of the quantized model  using the command:

```shell=
$ python eval_graph.py \
--graph ./_quant/quantize_eval_model.pb \
--input_node images_in \
--output_node dense_1/BiasAdd
```

And the expected results should be
 Top 1 accuracy with validation set: 0.9904
 Top 5 accuracy with validation set: 0.9999

The DNNC tool from DNNDK is used to deploy the inference kernel for the ZedBoard. The following command is used:

```shell=
dnnc-dpu1.4.0 \
--parser=tensorflow \
--frozen_pb=_quant/deploy_model.pb \
--dcf=zedboard.dcf \
--cpu_arch=arm32 \
--output_dir=_deploy \
--net_name=mnist \
--save_kernel \
--mode=normal
```

For simplicity, the Zedboard.dcf file is provided. Alternatively, it can be generated with the DLet tool from DNNDK, using the hardware hand-off (.hwh) file from the vivado project.

The generated inference kernel will be stored in "_deploy" folder in .elf file.

## Application build and run inference on the ZedBoard

Transfer to Zedboard the contents of the folder "mnist_zedboard_inference" via FTP.

Step 1: Install the application via "install.sh"\
Step 2: Build the model using the Makefile in samples/mnist\
Step 3: Run the generated executable to collect the result as mentioned in the paper

###### tags: `fpga` `dnndk` `zedboard`
