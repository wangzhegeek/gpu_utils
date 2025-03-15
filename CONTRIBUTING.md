# 贡献指南

感谢您对 GPU-Utils 项目的关注！我们欢迎各种形式的贡献，包括但不限于：

- 报告问题
- 提交功能请求
- 提交代码修复
- 提交新功能实现
- 改进文档

## 如何贡献

### 报告问题

如果您发现了 bug 或有改进建议，请通过 GitHub Issues 提交。提交问题时，请尽可能详细地描述：

1. 问题的具体表现
2. 复现步骤
3. 期望的行为
4. 您的环境信息（操作系统、CUDA 版本、GPU 型号等）

### 提交代码

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建一个 Pull Request

### 代码风格

- 对于 C++ 代码，请遵循 Google C++ 风格指南
- 保持代码整洁，添加必要的注释
- 为新功能编写测试

## 开发环境设置

1. 确保您已安装 CUDA Toolkit 12.4 或更高版本
2. 克隆仓库并编译：

```bash
git clone https://github.com/yourusername/gpu-utils.git
cd gpu-utils
cd gpu_monitors && make
cd ../gpu_test && make
```

## 测试

在提交代码之前，请确保：

1. 所有现有测试都能通过
2. 为新功能添加了适当的测试
3. 在不同的 GPU 型号上测试过（如果可能）

## 许可证

通过贡献您的代码，您同意您的贡献将在 MIT 许可证下发布。 