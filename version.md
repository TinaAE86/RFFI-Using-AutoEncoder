#### `LoraSigCtrl`
`init1` 构造函数1，使用带通信号制作数据集，不解调
`init2` 构造函数2，解调得到基带信号制作数据集

#### `PreDataset`
含有射频指纹的预处理数据集，是基带实信号，存储为`numpy`数组。`tensor`不直接存储而是使用时再创建
`v1` 数据本身不进行归一化
`v2` 会创建归一化的数据，采用min-max normalization, size=(Mn,N)
`v3` 同时包含I branch和Q branch，size=(Mn,2,N)

#### `DatasetGnt.gnt_pre_dataset()`
`v1` 生成`PreDataset`
`v2` 生成`PreDataset_v2`
`v3` 生成`PreDataset_v3`

#### `pre_dataset`
`v1` 实例化`PreDataset`，`gamma_r`=20，存储带通信号
`v2` `gamma_r`=20，存储基带信号，不含CFO
`v3` 实例化`PreDataset_v2`，`gamma_r`=20，存储基带信号，不含CFO
`v4` 实例化`PreDataset_v3`，`gamma_r`=20，存储基带信号，不含CFO

#### `rff_dataset`
`v1` `gamma_r`=20

#### `AE1d`
`v1` 带`relu`，将归一化层放在网络内
`v2` 网络内不含归一化，输出层含有`HardClamp`
`v3` 适配`PreDataset_v3`，输入初始含有双通道。取消输出层的`HardClamp`，并且取消输出层的`relu`防止下截断失真

#### `ae`
`v1` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=51, `lr`=1e-5，使用`AE1d_v2`
`v2` `batch_size`=8, `gamma_r`=20, `epoch_num`=50, `encode_channels`=4, `kernel_size`=101, `lr`=1e-6，使用`AE1d_v2`
~~`v3` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=203, `lr`=1e-6，使用`AE1d_v2`~~
~~`v4` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=51, `lr`=1e-6，使用`AE1d_v2`~~
~~`v5` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=101, `lr`=2e-6，使用`AE1d_v2`~~
~~`v6` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=101, `lr`=8e-7，使用`AE1d_v2`~~
`v7` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=101, `lr`=1e-6，使用`AE1d_v2`
`v8` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=51, `lr`=8e-6，使用`AE1d_v2`
`v9` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=2, `kernel_size`=51, `lr`=6e-6，使用`AE1d_v2`
`v10` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=31, `lr`=6e-6，使用`AE1d_v3`
`v11` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=15, `lr`=6e-6，使用`AE1d_v3`（输出层加了`relu`）
`v12` `batch_size`=8, `gamma_r`=20, `epoch_num`=100, `encode_channels`=4, `kernel_size`=15, `lr`=6e-6，使用`AE1d_v3`


#### `PreTrain.train()`
`v1` 使用`AE1d`
`v2` 使用`AE1d_v2`，loss选择归一化后的均方误差
`v3` 使用`AE1d_v3`，适配的数据初始为双通道