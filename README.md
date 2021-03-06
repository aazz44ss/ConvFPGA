# ConvFPGA
OpenCL based FPGA Convolution Accelerator with Systolic Array and Winograd

## Parallelism
Total MACs for convolution = Oh x Ow x Fo x Fi x Fh x Fw <br>
Parallelize M on Fo and N on Fi can increase M x N times on MAC/cycle

## Input Folding
We offsen set N as 16 (or larger), but the input feature map usally are pictures of 3 of Fi (RGB channel), causing the FPGA utilization only be 3/16 on first layer. <br>
By folding the input feature map to increase number of Fi in first layer to improve parallelism.

## Fixed Point 8 or Fixed Point 6
There are 1518 of 18x19 DSP in Altera 10 FPGA. <br>
It can do 759 FP32 multiplication per cycle because it needs 2 of DSP to calculate FP32 multiplication (fraction bits 23 larger than 18). <br>
It can do 1518 Fixed Point 8 multiplication per cycle (8 less than 18). <br>
It can do 3036 Fixed Point 6 multiplication per cycle by packing 2 of Fixed Point 6 into 18 Bits integer (FP6_0,6'0,FP6_1)

## Architecture of conv_core
### Systolic Array
Broadcasting data from DDR to MAC unit is not friendly for hardware layout, will cause very high latency (20-30 MHz clocks) in order to meet timing requirement. <br>
By using systolic array that pass data from DDR to PE(0) to PE(M-1), hardware layout can achieve low latency (130-160 MHz clocks) timing requirement. Can increase throughput by about 5 times compare to broadcast data from DDR. <br>
Each PE is consist of N multiplier and 4 shift register.

### Winograd
Use 1D winograd to increase throughput

![conv_core_arch](/documents/conv_core.png?raw=true "conv_core")

## Resource Usage on Intel Arria10 FPGA
![Resource Usage](/documents/resource.jpg?raw=true "Resource")
