# 第九章　精度（Precision）

> 译自 Alisa's Book of LLMs "Precision" 一节；底稿与约定见 [`05-scaling-laws.md`](05-scaling-laws.md) 页眉说明。

- **混合精度训练**（mixed precision）
  - 主权重（master weights）保持 FP32
  - 前向/反向传播用权重的 BF16 副本
  - 激活值用 BF16 计算
  - 梯度用 BF16 计算，**累加进 FP32**
    - 这避免了"往大权重上加小梯度"的丢数问题
    - BF16 在 0 附近精度高，能表示 0.0001 这样的小梯度，却表示不了更新后的权重 1.0001
    - 我们只需要累加后的梯度落到 FP32 里；单条梯度相对权重通常小得可以忽略
    - 每个参数的 `.grad` 与参数本身同类型，所以单条梯度会被转成 FP32
- 直觉
  - 矩阵乘法对舍入噪声很宽容，前向/反向用 BF16 没问题
  - 主权重必须留在 FP32，因为单条梯度太小
- 激活值比权重更难量化
- 精度选项
  - FP32：全精度（4 字节）
  - FP16：内存减半（2 字节）
  - BF16：和 FP16 一样省内存，但数值稳定性更好（动态范围大得多）
  - INT8：只有 FP32 的四分之一内存，需要量化（1 字节）
  - INT4：更小
- 数据加载
  - `memmap`（内存映射）避免把整个数据集一次性读进内存
- 用 BF16 加载模型
  - 用 HF 的 `.from_pretrained()` 加载时：
    - `torch_dtype=torch.bfloat16` 得到 BF16，**推荐做法**
      - 权重和激活都是 BF16
    - `load_in_8bit=True` 做量化，走 `LLM.int8()`
      - 权重 INT8，激活仍是 FP16
      - 省内存靠的是权重压缩
      - 因为激活精度更高，这不是真正的 INT8 全 INT8 计算
  - `model.half()` 或 `model.to(torch.bfloat16)` 把模型转成 FP16

    ```python
    model = MyModel()
    model.load_state_dict(torch.load('model.pt'))
    model = model.half()
    ```

  - 所有权重 BF16、所有运算 BF16
  - 实践上：推理全部用 BF16 没问题
- `torch.autocast()` 做自动混合精度
  - 权重保持 FP32，运算按算子选择性用 BF16 或 FP32
    - 它管理每个运算的精度，但不管理主权重
    - 比整个模型加载成 BF16 更占内存，但可能更稳定
  - 矩阵乘法用 FP16（容得下低精度），softmax 用 FP32（数值稳定需要精度），layer norm 用 FP32（归约运算需要精度）
  - 适用场景：模型是 FP32 且不方便整体转换，或纯 BF16 出现数值问题时
  - `dtype` 指定"低精度"那一档

  ```python
  with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      output = model(x)
  ```

- 用 `bitsandbytes`：把 `nn.Linear` 换成 `bnb.nn.Linear8bitLt` 或 `bnb.nn.Linear4bit`

  ```python
  def replace_linear_with_8bit(model):
      """把所有 nn.Linear 替换成 bnb.nn.Linear8bitLt"""
      for name, child in model.named_children():
          if isinstance(child, nn.Linear):
              # 构造量化替代品
              new_layer = bnb.nn.Linear8bitLt(
                  child.in_features,
                  child.out_features,
                  bias=child.bias is not None,
                  has_fp16_weights=False,
              )
              # 拷贝权重（搬到 CUDA 时会完成量化）
              new_layer.weight = bnb.nn.Int8Params(
                  child.weight.data,
                  requires_grad=False,
              )
              if child.bias is not None:
                  new_layer.bias = nn.Parameter(child.bias.data)

              setattr(model, name, new_layer)
          else:
              # 递归处理子模块
              replace_linear_with_8bit(child)

      return model

  # 用法
  model = MyTransformer()
  model.load_state_dict(torch.load('model.pt'))
  model = replace_linear_with_8bit(model)
  model = model.to('cuda')  # 量化在这一步发生
  ```
