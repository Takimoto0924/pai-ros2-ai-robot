# ROS 2 × Gazebo 人物探索ロボット

ROS 2 Jazzy と Gazebo Harmonic を用いた、シミュレーション上の人物探索ロボットです。
学習済み **YOLO11n** でカメラ画像から人物を検出し、**LiDAR** で周囲の障害物を確認しながら人物への接近を目指します。

## 動作環境

- Ubuntu 24.04 / ROS 2 Jazzy
- Gazebo Harmonic
- Python 3（`rclpy`、`cv_bridge`、`ultralytics`、`torch`、`opencv-python` など）

## ファイル構成

```text
pai_final/
├── src/
│   ├── controller.py               # 人物探索・移動制御
│   ├── create_world.py             # ワールド生成
│   └── create_camera_world.py      # カメラ付きワールド生成
├── worlds/
│   ├── camera_world.sdf
│   └── object_world.sdf
├── docs/
│   └── result.log                  # 実行ログ
├── yolo11n.pt                      # 学習済みYOLOモデル
└── README.md
```

## 実行方法

**1. Gazeboを起動**

```bash
gz sim -r -v 4 worlds/object_world.sdf
```

**2. 別のターミナルでROS 2 Bridgeを起動**

```bash
source /opt/ros/jazzy/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  '/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist' \
  '/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' \
  '/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan' \
  '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image'
```

**3. さらに別のターミナルで制御プログラムを実行**

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python src/controller.py
```

※ コマンドはリポジトリのルートディレクトリで実行してください。実行前に必要なPythonパッケージをインストールしてください。

## 主な機能

- YOLO11nによる人物検出
- カメラ画像とLiDAR距離を利用した人物への接近
- LiDARに基づく障害物検出・迂回
- 人物を見失った場合の旋回探索と再探索

※ 実験用の制御であり、すべての障害物配置での到達を保証するものではありません。
