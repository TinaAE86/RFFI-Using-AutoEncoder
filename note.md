#### `SigCtrlBase`
信号控制器基类
`t_arr` $N$点时间序列，方便计算和绘图
`a_arr` $N$点基带复信号
`rff_arr` $N$点带通信号，含有调制射频指纹

#### `PreDataset`
预训练数据集，数据结构为结构体{`t_arr`, `s`, `r`}