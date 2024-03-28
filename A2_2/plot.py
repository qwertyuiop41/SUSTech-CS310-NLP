import matplotlib.pyplot as plt

num_configs = 18  # 配置的总数
log_files = ["A2_pdf.log"]  # 日志文件列表

for config_index in range(1, num_configs+1):
    config_losses = []  # 损失数据列表

    # 解析对应配置的日志文件
    log_file="A2_pdf.log"
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if f"Config {config_index}:" in line:
                break
        else:
            continue

    # 提取配置的损失数据
    for line in lines:
        if "Loss:" in line:
            loss = float(line.split("Loss:")[1])
            config_losses.append(loss)

    # 绘制折线图
    plt.plot(range(1, len(config_losses) + 1), config_losses, label=f'Config {config_index}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss for Each Config')
plt.legend()
plt.show()