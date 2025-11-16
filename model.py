import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort

class DynamicWeightedFusion(nn.Module):
    """
    动态权重融合模块
    输入: P (自上而下特征), C (自下而上特征)
    输出: 加权融合后的特征
    """

    def __init__(self, channels, reduction=16):
        super(DynamicWeightedFusion, self).__init__()
        self.channels = channels

        # 权重生成网络 - 轻量化设计
        self.weight_net = nn.Sequential(
            # 第一步: 通道压缩
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),

            # 第二步: 空间信息压缩 (全局平均池化)
            # 第三步: 全连接层生成权重
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels // reduction, 2, 1, bias=True),
            nn.Sigmoid()  # 输出[0,1]范围的权重
        )

    def forward(self, p, c):
        """
        Args:
            p: 自上而下特征 [B, C, H, W]
            c: 自下而上特征 [B, C, H, W]
        Returns:
            fused: 动态加权融合后的特征 [B, C, H, W]
            weights: 生成的权重 [B, 2, 1, 1]
        """
        # 1. 特征对齐 (确保尺寸一致)
        print('='*50)
        print('p', p.shape)
        print('c', c.shape)
        if p.size() != c.size():
            c = F.interpolate(c, size=p.shape[2:], mode='nearest')
        print('p', p.shape)
        print('c', c.shape)

        # 2. 拼接特征用于权重计算
        feature_cat = torch.cat([p, c], dim=1)  # [B, 2C, H, W]
        print('feature_cat', feature_cat.shape)
        # 3. 动态生成权重
        weights = self.weight_net(feature_cat)  # [B, 2, 1, 1]
        w_p, w_c = weights[:, 0:1], weights[:, 1:2]  # 分别对应P和C的权重
        print('w_p', w_p.shape)
        print('w_c', w_p.shape)

        # 4. 加权融合
        fused = w_p * p + w_c * c
        print('='*50)

        return fused, weights


class BottomUpDynamicFusionBlock(nn.Module):
    """
    完整的自下而上动态融合模块
    集成到YOLOv7的FPN结构中
    """

    def __init__(self, in_channels_list, out_channels):
        super(BottomUpDynamicFusionBlock, self).__init__()

        # 自下而上的卷积通路 (用于特征转换)
        self.bottom_up_conv = nn.Conv2d(in_channels_list[0], out_channels, 3, padding=1)

        # 横向连接的卷积 (如果需要调整通道数)
        self.lateral_conv = nn.Conv2d(in_channels_list[1], out_channels, 1)

        # 动态权重融合模块
        self.dynamic_fusion = DynamicWeightedFusion(out_channels)

        # 融合后的输出卷积
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, bottom_up_feat, lateral_feat):
        """
        Args:
            bottom_up_feat: 来自底层的特征 [B, C1, H1, W1]
            lateral_feat: 横向连接的特征 [B, C2, H2, W2]
        """
        # 1. 特征转换
        bottom_up_processed = self.bottom_up_conv(bottom_up_feat)
        lateral_processed = self.lateral_conv(lateral_feat)
        print('-'*25)
        print('bottom_up_feat', bottom_up_feat.shape)
        print('bottom_up_processed', bottom_up_processed.shape)
        print('lateral_feat', lateral_feat.shape)
        print('lateral_processed', lateral_processed.shape)

        # 2. 动态权重融合
        fused_feat, weights = self.dynamic_fusion(lateral_processed, bottom_up_processed)
        print('fused_feat', fused_feat.shape)

        # 3. 输出处理
        output = self.output_conv(fused_feat)
        print('output', output.shape)

        print('-'*25)

        return output, weights


class EnhancedYOLOv7FPN(nn.Module):
    """
    增强的YOLOv7 FPN，包含自下而上的动态融合通路
    """

    def __init__(self, backbone_channels=[512, 256, 128], neck_channels=256):
        super(EnhancedYOLOv7FPN, self).__init__()

        # 标准自上而下通路 (YOLOv7原有)
        self.top_down_convs = nn.ModuleList([
            nn.Conv2d(backbone_channels[0], neck_channels, 1),
            nn.Conv2d(backbone_channels[1], neck_channels, 1),
            nn.Conv2d(backbone_channels[2], neck_channels, 1)
        ])

        # 新增: 自下而上的动态融合通路
        self.bottom_up_fusions = nn.ModuleList([
            # P3 -> P4 融合
            BottomUpDynamicFusionBlock(
                in_channels_list=[neck_channels, neck_channels],  # [P3, P4]
                out_channels=neck_channels
            ),
            # P4 -> P5 融合
            BottomUpDynamicFusionBlock(
                in_channels_list=[neck_channels, neck_channels],  # [P4, P5]
                out_channels=neck_channels
            )
        ])

    def forward(self, backbone_features):
        """
        Args:
            backbone_features: 主干网络输出的多尺度特征 [C3, C4, C5]
        """
        # C3, C4, C5 对应不同尺度的特征
        c3, c4, c5 = backbone_features
        # === 标准自上而下通路 ===
        # P5: 最高层特征
        print(c5.shape)
        p5 = self.top_down_convs[0](c5)
        print(p5.shape)

        # P4: 上采样P5并与C4融合
        p5_upsampled = F.interpolate(p5, scale_factor=2, mode='nearest')
        print('p5_upsampled', p5_upsampled.shape)
        p4_input = self.top_down_convs[1](c4) + p5_upsampled
        print('p4_input', p4_input.shape)


        # P3: 上采样P4并与C3融合
        p4_upsampled = F.interpolate(p4_input, scale_factor=2, mode='nearest')
        print('p4_upsampled', p4_upsampled.shape)

        p3 = self.top_down_convs[2](c3) + p4_upsampled
        print('p3', p3.shape)

        # === 新增: 自下而上的动态融合通路 ===
        # 第一级融合: P3 -> P4
        p4_enhanced, weights_1 = self.bottom_up_fusions[0](p3, p4_input)
        print('p4_enhanced', p4_enhanced.shape)

        # 第二级融合: P4 -> P5
        p5_enhanced, weights_2 = self.bottom_up_fusions[1](p4_enhanced, p5)
        print('p5_enhanced', p5_enhanced.shape)

        return {
            'p3': p3,  # 原始自上而下特征
            'p4': p4_enhanced,  # 增强后的特征 (包含自下而上信息)
            'p5': p5_enhanced,  # 增强后的特征
            'fusion_weights': [weights_1, weights_2]  # 保存权重用于分析
        }





class ExportReadyEnhancedYOLOv7FPN(nn.Module):
    """
    专门为ONNX导出优化的版本
    将三个输入明确分开，避免动态输入结构
    """

    def __init__(self, backbone_channels=[512, 256, 128], neck_channels=256):
        super(ExportReadyEnhancedYOLOv7FPN, self).__init__()

        # 保持原有的网络结构不变
        self.top_down_convs = nn.ModuleList([
            nn.Conv2d(backbone_channels[0], neck_channels, 1),
            nn.Conv2d(backbone_channels[1], neck_channels, 1),
            nn.Conv2d(backbone_channels[2], neck_channels, 1)
        ])

        self.bottom_up_fusions = nn.ModuleList([
            BottomUpDynamicFusionBlock(
                in_channels_list=[neck_channels, neck_channels],
                out_channels=neck_channels
            ),
            BottomUpDynamicFusionBlock(
                in_channels_list=[neck_channels, neck_channels],
                out_channels=neck_channels
            )
        ])

    def forward(self, c3, c4, c5):
        """
        明确分离三个输入参数，便于ONNX导出
        Args:
            c3: [B, 128, H3, W3]  高层特征
            c4: [B, 256, H4, W4]  中层特征
            c5: [B, 512, H5, W5]  底层特征
        """
        # === 标准自上而下通路 ===
        # P5: 最高层特征
        p5 = self.top_down_convs[0](c5)

        # P4: 上采样P5并与C4融合
        p5_upsampled = F.interpolate(p5, scale_factor=2, mode='nearest')
        p4_input = self.top_down_convs[1](c4) + p5_upsampled

        # P3: 上采样P4并与C3融合
        p4_upsampled = F.interpolate(p4_input, scale_factor=2, mode='nearest')
        p3 = self.top_down_convs[2](c3) + p4_upsampled

        # === 自下而上的动态融合通路 ===
        p4_enhanced, weights_1 = self.bottom_up_fusions[0](p3, p4_input)
        p5_enhanced, weights_2 = self.bottom_up_fusions[1](p4_enhanced, p5)

        # 只返回必要的检测特征，避免返回复杂结构
        return p3, p4_enhanced, p5_enhanced


def export_to_onnx():
    """
    导出三输入EnhancedYOLOv7FPN到ONNX
    """
    # 1. 创建模型实例
    model = ExportReadyEnhancedYOLOv7FPN()
    model.eval()

    # 2. 创建三个不同尺度的示例输入
    batch_size = 1
    # 假设输入图像为640x640，特征图尺寸按stride递减
    dummy_c3 = torch.randn(batch_size, 128, 80, 80)  # stride=8
    dummy_c4 = torch.randn(batch_size, 256, 40, 40)  # stride=16
    dummy_c5 = torch.randn(batch_size, 512, 20, 20)  # stride=32

    # 3. 导出ONNX - 关键步骤
    torch.onnx.export(
        model,
        (dummy_c3, dummy_c4, dummy_c5),  # 注意：三个输入作为元组传递
        "enhanced_yolov7_fpn.onnx",
        input_names=['input_c3', 'input_c4', 'input_c5'],  # 三个输入名称
        output_names=['output_p3', 'output_p4', 'output_p5'],  # 三个输出名称
        dynamic_axes={
            'input_c3': {0: 'batch_size', 2: 'height_c3', 3: 'width_c3'},
            'input_c4': {0: 'batch_size', 2: 'height_c4', 3: 'width_c4'},
            'input_c5': {0: 'batch_size', 2: 'height_c5', 3: 'width_c5'},
            'output_p3': {0: 'batch_size', 2: 'height_p3', 3: 'width_p3'},
            'output_p4': {0: 'batch_size', 2: 'height_p4', 3: 'width_p4'},
            'output_p5': {0: 'batch_size', 2: 'height_p5', 3: 'width_p5'}
        },
        opset_version=13,  # 使用较高的opset以支持更多操作
        do_constant_folding=True,
        verbose=True
    )

    print("ONNX导出成功！")


def verify_onnx_model():
    """
    验证导出的ONNX模型
    """
    # 1. 加载ONNX模型
    onnx_model = onnx.load("enhanced_yolov7_fpn.onnx")
    onnx.checker.check_model(onnx_model)

    # 2. 创建ONNX Runtime会话
    ort_session = ort.InferenceSession("enhanced_yolov7_fpn.onnx")

    # 3. 准备测试数据
    batch_size = 1
    test_c3 = torch.randn(batch_size, 128, 80, 80).numpy()
    test_c4 = torch.randn(batch_size, 256, 40, 40).numpy()
    test_c5 = torch.randn(batch_size, 512, 20, 20).numpy()

    # 4. 运行推理
    ort_inputs = {
        'input_c3': test_c3,
        'input_c4': test_c4,
        'input_c5': test_c5
    }

    ort_outputs = ort_session.run(None, ort_inputs)

    print(f"ONNX模型验证成功！")
    print(f"输出数量: {len(ort_outputs)}")
    print(f"P3形状: {ort_outputs[0].shape}")
    print(f"P4形状: {ort_outputs[1].shape}")
    print(f"P5形状: {ort_outputs[2].shape}")


# # 运行导出和验证
# if __name__ == "__main__":
#     export_to_onnx()
#     verify_onnx_model()



if __name__ == '__main__':
    # 创建一个示例的输入数据
    input_data3 = torch.randn(1, 512, 10, 10)
    input_data2 = torch.randn(1, 256, 20, 20)
    input_data1 = torch.randn(1, 128, 40, 40)
    input_data = [input_data1, input_data2, input_data3]
    # 创建一个EnhancedYOLOv7FPN实例
    model = EnhancedYOLOv7FPN()
    # 前向传播
    output = model(input_data)

    # # 输出结果
    # print(output.keys())
    # print(output['p3'].shape)
    # print(output['p4'].shape)

