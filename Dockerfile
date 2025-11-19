FROM registry.cn-hangzhou.aliyuncs.com/anyshu/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# 设置工作目录
WORKDIR /app

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
RUN apt update && \
    apt install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt ./

# 设置国内 pip 镜像并安装依赖
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY main.py service.py ./

# 暴露端口
EXPOSE 8000

# 设置默认环境变量
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1

# 启动命令（根据你的实际启动方式调整）
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]