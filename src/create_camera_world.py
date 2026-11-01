#!/usr/bin/env python3
"""Gazebo Harmonic + ROS 2 Jazzy: 人物の360°探索、接近、障害物回避。

トピック:
  /camera/image                  sensor_msgs/Image (Gazebo -> ROS)
  /lidar                         sensor_msgs/LaserScan (Gazebo -> ROS)
  /model/vehicle_blue/odometry   nav_msgs/Odometry (Gazebo -> ROS)
  /model/vehicle_blue/cmd_vel    geometry_msgs/Twist (ROS -> Gazebo)

人物検出: 学習済みYOLO11n（COCO person）をCPUで実行。
距離判定: RGB検出枠の高さと前方LiDARを併用した簡易的な判定。
重要: LiDAR反射の対象が人物であることは保証できない。
"""

import math
import threading
import time

import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from ultralytics import YOLO


def wrap_angle(angle):
    """角度を[-π, π]に正規化する。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class PersonNavigationController(Node):
    def __init__(self):
        super().__init__("person_navigation_controller")

        torch.set_num_threads(2)
        self.model = YOLO("yolo11n.pt")
        self.cv_bridge = CvBridge()

        # ===== 実験に合わせて調整できる値 =====
        self.search_speed = 0.25           # [rad/s] SCANの旋回速度
        self.turn_max_speed = 1.0          # [rad/s] 人物へ向くときの速度上限
        self.avoid_turn_speed = 1.0        # [rad/s] 迂回時の旋回速度上限
        self.approach_speed = 1.50         # [m/s] 人物への最大前進速度
        self.detour_speed = 1.50           # [m/s] 迂回時の前進速度
        self.explore_speed = 1.50          # [m/s] 未発見時の探索移動速度

        self.obstacle_trigger = 1.65      # [m] 前方障害物を回避し始める距離
        self.emergency_distance = 0.10    # [m] 最優先の緊急停止
        self.detour_stop_distance = 1.05  # [m] 迂回時に前進を中止する距離
        self.target_distance = 0.50       # [m] 人物接近完了の補助条件
        # 人物が大きく見えるだけでは、LiDARが人物に反射したとは断定できない。
        # obstacle-free実験では近距離の「高さ比 × 前方距離」が約0.85〜1.1。
        # medium実験で手前の障害物に近づくと0.51程度まで減少した。
        # 両者を切り分ける実験用の保守的な基準（万能な人物距離推定ではない）。
        # 人物の見かけの高さに基づく遮蔽物判定（速度設定は変更しない）
        # d_person_est = height_distance_constant / height_ratio
        # この値は障害物なしの実験で較正すること。暫定値1.0 [m]。
        self.height_distance_constant = 1.0
        self.obstacle_distance_fraction = 0.70
        self.person_lock_height_ratio = 0.52
        self.arrival_height_ratio = 0.58
        self.stop_height_ratio = 0.73      # 従来設定の記録用
        self.min_person_streak = 3         # 連続3回の人物検出を要する
        self.side_distance = 2.4          # [m] 横に回り込む距離
        self.pass_distance = 4.4          # [m] 障害物の脇を通る距離
        self.explore_distance = 1.4       # [m] 発見できない場合の場所変更量
        self.max_avoid_count = 6           # 複数障害物用の上限
        self.max_explore_count = 5
        self.max_duration = 360.0         # [s] 試験の時間上限
        self.person_lost_grace = 1.5      # [s] 人物見失い時の猶予

        self.cmd_pub = self.create_publisher(
            Twist, "/model/vehicle_blue/cmd_vel", 10
        )
        self.create_subscription(
            Image, "/camera/image", self.image_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan, "/lidar", self.lidar_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, "/model/vehicle_blue/odometry", self.odom_callback, 10
        )

        # センサーはROSコールバック側で更新する。
        self.pose = None  # (x, y, yaw)
        self.front = None
        self.left = None
        self.right = None
        self.last_image = None
        self.last_scan = None
        self.last_odom = None

        # 推論処理は別スレッドで実行し、ROS制御タイマーを塞がない。
        self.lock = threading.Lock()
        self.pending_frame = None
        self.last_queued = 0.0
        self.person = None
        self.last_inference = None
        self.last_seen = None
        self.detection_streak = 0
        self.inference_error = None
        self.person_path_confirmed = False
        self.person_last_front = None
        self.person_last_pose = None
        self.worker_stop = threading.Event()
        self.worker = threading.Thread(
            target=self.inference_loop, daemon=True
        )
        self.worker.start()

        # SCAN -> APPROACH -> AVOID_* / SCAN -> EXPLORE_* -> SCAN
        self.state = "SCAN"
        self.state_enter_time = time.monotonic()
        self.scan_prev_yaw = None
        self.scan_accumulated = 0.0
        self.target_yaw = None
        self.original_yaw = None
        self.segment_start = None
        self.avoid_direction = 0
        self.avoid_count = 0
        self.explore_count = 0
        self.finished = False
        self.start_time = time.monotonic()
        self.last_log = 0.0

        self.timer = self.create_timer(0.1, self.control)
        self.get_logger().info(
            "[SCAN] 起動：まず周囲を旋回して人物を探索します"
        )

    # --------------------- センサーデータ取得 ---------------------

    def image_callback(self, msg):
        now = time.monotonic()
        self.last_image = now
        if now - self.last_queued < 0.65:
            return
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            with self.lock:
                # YOLOの処理が追いつかない場合は最新フレームだけを残す。
                self.pending_frame = frame
            self.last_queued = now
        except Exception as exc:
            self.get_logger().error(f"画像変換エラー: {exc}")

    def inference_loop(self):
        while not self.worker_stop.is_set():
            with self.lock:
                frame = self.pending_frame
                self.pending_frame = None
            if frame is None:
                self.worker_stop.wait(0.04)
                continue
            try:
                result = self.model.predict(
                    source=frame, classes=[0], conf=0.35,
                    imgsz=416, device="cpu", verbose=False
                )[0]
                detected = None
                if len(result.boxes) > 0:
                    box = max(
                        result.boxes,
                        key=lambda item: float(item.conf[0])
                    )
                    x1, y1, x2, y2 = map(
                        float, box.xyxy[0].tolist()
                    )
                    height, width = frame.shape[:2]
                    detected = {
                        "confidence": float(box.conf[0]),
                        "error": ((x1 + x2) / 2 - width / 2) / (width / 2),
                        "height_ratio": (y2 - y1) / height,
                    }
                now = time.monotonic()
                with self.lock:
                    self.person = detected
                    self.last_inference = now
                    if detected is None:
                        self.detection_streak = 0
                    else:
                        self.detection_streak += 1
                        self.last_seen = now
            except Exception as exc:
                with self.lock:
                    self.inference_error = str(exc)
                return

    def lidar_callback(self, msg):
        sectors = {"front": [], "left": [], "right": []}
        for index, measured in enumerate(msg.ranges):
            deg = math.degrees(
                msg.angle_min + index * msg.angle_increment
            )
            if math.isinf(measured) and measured > 0:
                distance = msg.range_max
            elif (math.isfinite(measured)
                  and msg.range_min <= measured <= msg.range_max):
                distance = measured
            else:
                continue
            if -20 <= deg <= 20:
                sectors["front"].append(distance)
            if 45 <= deg <= 125:
                sectors["left"].append(distance)
            if -125 <= deg <= -45:
                sectors["right"].append(distance)
        self.front = min(sectors["front"], default=None)
        self.left = min(sectors["left"], default=None)
        self.right = min(sectors["right"], default=None)
        self.last_scan = time.monotonic()

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )
        self.pose = (p.x, p.y, yaw)
        self.last_odom = time.monotonic()

    # --------------------- 基本制御 ---------------------

    def move(self, linear=0.0, angular=0.0):
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)
        self.cmd_pub.publish(command)

    def finish(self, reason):
        self.finished = True
        self.move()
        self.get_logger().info(f"[STOP] {reason}")

    def log_periodic(self, message, interval=2.5):
        now = time.monotonic()
        if now - self.last_log >= interval:
            self.get_logger().info(message)
            self.last_log = now

    def change_state(self, state, message):
        self.move()
        self.state = state
        self.state_enter_time = time.monotonic()
        self.get_logger().info(f"[{state}] {message}")

    def start_scan(self, message):
        # yawの差分を積算する。単なる開始角との差だと360°は測れない。
        self.person_path_confirmed = False
        self.person_last_front = None
        self.person_last_pose = None
        self.scan_prev_yaw = self.pose[2]
        self.scan_accumulated = 0.0
        self.change_state("SCAN", message)

    def rotate_to(self, target):
        error = wrap_angle(target - self.pose[2])
        if abs(error) < 0.08:
            self.move()
            return True
        speed = clamp(
            1.0 * error, -self.avoid_turn_speed, self.avoid_turn_speed
        )
        if abs(speed) < 0.11:
            speed = math.copysign(0.11, error)
        self.move(0.0, speed)
        return False

    def distance_from_start(self):
        return math.hypot(
            self.pose[0] - self.segment_start[0],
            self.pose[1] - self.segment_start[1]
        )

    def follow_heading(self, speed):
        error = wrap_angle(self.target_yaw - self.pose[2])
        turn = clamp(0.8 * error, -0.16, 0.16)
        self.move(speed, turn)

    # --------------------- 周囲の360°探索 ---------------------

    def run_scan(self, person, streak, detected_at):
        if self.scan_prev_yaw is None:
            self.scan_prev_yaw = self.pose[2]

        # 2フレーム連続で人物を認識したら、その場で探索を終了。
        # 初回でも人物が見つかれば、360°回り切る必要はない。
        if (person is not None and streak >= 2
                and detected_at >= self.state_enter_time):
            self.change_state(
                "APPROACH",
                f"人物を発見: 信頼度={person['confidence']:.2f}"
            )
            return

        current_yaw = self.pose[2]
        delta_yaw = wrap_angle(current_yaw - self.scan_prev_yaw)
        self.scan_accumulated += abs(delta_yaw)
        self.scan_prev_yaw = current_yaw

        if self.scan_accumulated >= 2 * math.pi - 0.10:
            self.move()
            self.start_explore()
            return

        self.move(0.0, self.search_speed)
        self.log_periodic(
            f"[SCAN] 周囲を探索中："
            f"{math.degrees(self.scan_accumulated):.0f}° / 360°"
        )

    # --------------------- 未発見時の位置変更 ---------------------

    def select_direction(self, allow_forward):
        """前方・左・右の空き具合から進行方向を選択。"""
        # 左右が接近している場合は、交互に選んで探索位置を変える。
        # あくまで探索方針であって、左右の優劣が計測できたわけではない。
        if (allow_forward and self.front >= 2.3):
            return 0
        if max(self.left, self.right) < 1.8:
            return None
        if abs(self.left - self.right) < 0.20:
            return 1 if self.explore_count % 2 == 0 else -1
        return 1 if self.left > self.right else -1

    def start_explore(self):
        if self.explore_count >= self.max_explore_count:
            self.finish("360°探索と場所変更の上限に達しました")
            return
        direction = self.select_direction(allow_forward=True)
        if direction is None:
            self.finish("人物未発見、周囲に探索移動できる空間がありません")
            return
        self.explore_count += 1
        self.target_yaw = wrap_angle(
            self.pose[2] + direction * math.pi / 2
        )
        label = {0: "前方", 1: "左側", -1: "右側"}[direction]
        self.change_state(
            "EXPLORE_TURN",
            f"360°で人物を発見できず、{label}へ場所を変更 "
            f"({self.explore_count}/{self.max_explore_count}回)"
        )

    def run_explore(self):
        if self.state == "EXPLORE_TURN":
            if self.rotate_to(self.target_yaw):
                self.segment_start = self.pose[:2]
                self.change_state(
                    "EXPLORE_MOVE", "低速で位置を変更します"
                )
            return

        if self.state == "EXPLORE_MOVE":
            traveled = self.distance_from_start()
            if self.front <= 1.30:
                if traveled >= 0.35:
                    self.start_scan(
                        "探索移動中に障害物を検出。新しい位置で再探索"
                    )
                else:
                    self.finish(
                        f"探索経路が塞がれています: 前方={self.front:.2f}m"
                    )
                return
            if traveled >= self.explore_distance:
                self.start_scan("位置変更完了。再度360°探索します")
                return
            self.follow_heading(self.explore_speed)
            self.log_periodic(
                f"[EXPLORE_MOVE] {traveled:.2f}/"
                f"{self.explore_distance:.2f}m"
            )

    # --------------------- 障害物回避 ---------------------

    def start_avoidance(self):
        if self.avoid_count >= self.max_avoid_count:
            self.finish("迂回回数の上限に達しました")
            return
        direction = self.select_direction(allow_forward=False)
        if direction is None:
            self.finish("障害物の左右に通行余裕がなく停止しました")
            return
        self.person_path_confirmed = False
        self.person_last_front = None
        self.person_last_pose = None
        self.avoid_count += 1
        self.avoid_direction = direction
        self.original_yaw = self.pose[2]
        self.target_yaw = wrap_angle(
            self.original_yaw + direction * math.pi / 2
        )
        label = "左" if direction == 1 else "右"
        self.change_state(
            "AVOID_TURN_SIDE",
            f"前方={self.front:.2f}m、{label}へ迂回 "
            f"({self.avoid_count}/{self.max_avoid_count}回)"
        )

    def run_avoidance(self):
        if self.state == "AVOID_TURN_SIDE":
            if self.rotate_to(self.target_yaw):
                self.segment_start = self.pose[:2]
                self.change_state(
                    "AVOID_SIDE", "障害物の横へ回り込みます"
                )
            return

        if self.state == "AVOID_SIDE":
            if self.front <= self.detour_stop_distance:
                self.move()
                traveled = self.distance_from_start()
                if traveled >= 0.35:
                    # 袋小路に遭遇したら無理に前進しない。
                    # 現在位置で人物を再探索し、別の経路を探す。
                    self.start_scan(
                        f"横迂回が塞がれました: 前方={self.front:.2f}m。"
                        "現在位置で再探索します"
                    )
                else:
                    # ほとんど動けないときは安全のため停止する。
                    self.finish(
                        f"横方向の迂回経路に余裕がありません: "
                        f"前方={self.front:.2f}m"
                    )
                return
            if self.distance_from_start() >= self.side_distance:
                self.target_yaw = self.original_yaw
                self.change_state(
                    "AVOID_TURN_FORWARD", "元の進行方向へ向き直ります"
                )
            else:
                self.follow_heading(self.detour_speed)
            return

        if self.state == "AVOID_TURN_FORWARD":
            if self.rotate_to(self.target_yaw):
                self.segment_start = self.pose[:2]
                self.change_state(
                    "AVOID_PASS", "障害物の側方を前進します"
                )
            return

        if self.state == "AVOID_PASS":
            if self.front <= self.detour_stop_distance:
                self.move()
                if self.distance_from_start() >= 0.35:
                    self.start_scan(
                        f"迂回前進中に別の障害物を検出: "
                        f"前方={self.front:.2f}m。再探索します"
                    )
                else:
                    self.finish(
                        f"迂回前進の経路がすぐに塞がれました: "
                        f"前方={self.front:.2f}m"
                    )
                return
            if self.distance_from_start() >= self.pass_distance:
                self.start_scan("迂回完了。人物を再探索します")
            else:
                self.follow_heading(self.detour_speed)

    # --------------------- 人物への接近 ---------------------

    def estimate_person_distance(self, height_ratio):
        """見かけの人物の高さから距離を概算する（同じActor・画角専用）。

        近距離では全身が画角からはみ出すため、この概算値は使用しない。
        補正係数は、障害物のない環境で実測して調整する。
        """
        return self.height_distance_constant / max(height_ratio, 0.05)

    def run_approach(self, person, last_seen, now):
        if person is None:
            self.move()
            if (last_seen is None
                    or now - last_seen > self.person_lost_grace):
                self.start_scan("人物を見失いました。360°再探索します")
            return

        error = person["error"]
        ratio = person["height_ratio"]
        aligned = abs(error) <= 0.18
        enough_detections = (
            self.detection_streak >= self.min_person_streak
        )
        estimated_person_distance = self.estimate_person_distance(ratio)

        # 人物の見かけの距離より、LiDARが明らかに手前を指すなら遮蔽物の疑い。
        # この規則は未確認の進路でのみ使い、近距離での画像切れによる
        # 「人物=障害物」の誤判定を避ける。
        obstacle_ahead = (
            self.front <= self.obstacle_trigger
            and self.front < (
                self.obstacle_distance_fraction * estimated_person_distance
            )
        )

        # 遠方で画像高さとLiDAR距離の整合が取れ、かつ複数回検出したら、
        # 進行方向に人物がいる見込みとして追跡状態を保持する。
        # 以後は画像高さが1に近づいても追跡を継続できる。
        if (aligned and enough_detections
                and ratio >= self.person_lock_height_ratio
                and not obstacle_ahead
                and self.front <= 2.5):
            if not self.person_path_confirmed:
                self.get_logger().info(
                    f"[APPROACH] 人物への進路を暫定確認: "
                    f"前方={self.front:.2f}m, 高さ比={ratio:.2f}, "
                    f"高さ由来の推定距離={estimated_person_distance:.2f}m"
                )
            self.person_path_confirmed = True
            self.person_last_front = self.front
            self.person_last_pose = self.pose[:2]

        # 進路を確認した後でも、LiDAR距離が実移動量に比べて突然
        # 大幅に短くなれば新たな障害物が出現した疑いとして追跡を解除。
        if (self.person_path_confirmed
                and self.person_last_front is not None
                and self.person_last_pose is not None):
            displacement = math.hypot(
                self.pose[0] - self.person_last_pose[0],
                self.pose[1] - self.person_last_pose[1]
            )
            sudden_drop = (
                self.person_last_front - self.front > displacement + 0.7
            )
            if sudden_drop:
                self.person_path_confirmed = False
                obstacle_ahead = True
                self.get_logger().warn(
                    "[APPROACH] 前方距離の急変を検知。進路を再評価します"
                )

        # 比較用の直前観測を更新。速度設定は変更しない。
        self.person_last_front = self.front
        self.person_last_pose = self.pose[:2]

        # 方向がまだ合っていないなら、直進せずその場で旋回。
        if not aligned:
            turn = clamp(
                -0.35 * error,
                -self.turn_max_speed, self.turn_max_speed
            )
            self.move(0.0, turn)
            self.log_periodic(
                f"[APPROACH] 方向調整: 左右誤差={error:.2f}"
            )
            return

        # 進路が未確認で、手前に障害物がある場合は迂回。
        if obstacle_ahead and not self.person_path_confirmed:
            self.get_logger().info(
                f"[APPROACH] 障害物の可能性: "
                f"LiDAR={self.front:.2f}m < "
                f"{self.obstacle_distance_fraction:.2f}×"
                f"推定人物距離={estimated_person_distance:.2f}m -> 迂回"
            )
            self.start_avoidance()
            return

        # 近距離ではK/h近似が破綻するため、遠距離で確認した追跡状態を
        # 維持しつつ、YOLOが連続して人物を認識している場合だけ到達扱い。
        # target_distanceはユーザー設定0.30mを維持する。
        if (self.person_path_confirmed and enough_detections
                and ratio >= self.arrival_height_ratio
                and self.front <= self.target_distance):
            self.finish(
                f"人物への接近完了（推定）: 高さ比={ratio:.2f}, "
                f"前方LiDAR={self.front:.2f}m "
                "（対象が人物である保証はありません）"
            )
            return

        # 突然の遮蔽物が疑われない限り、人物へ接近する。
        speed = (
            1.0 if self.front > 3.0 else
            0.50 if self.front > 1.8 else 0.10
        )
        speed = min(speed, self.approach_speed)
        turn = clamp(-0.35 * error, -0.12, 0.12)
        self.move(speed, turn)
        self.log_periodic(
            f"[APPROACH] 人物へ接近: "
            f"信頼度={person['confidence']:.2f}, "
            f"高さ比={ratio:.2f}, 前方={self.front:.2f}m, "
            f"推定人物距離={estimated_person_distance:.2f}m, "
            f"追跡確認={self.person_path_confirmed}, 速度={speed:.2f}m/s"
        )

    # --------------------- 制御ループ ---------------------

    def control(self):
        if self.finished:
            return
        now = time.monotonic()
        if now - self.start_time >= self.max_duration:
            self.finish("試験時間上限のため停止しました")
            return
        if self.cmd_pub.get_subscription_count() == 0:
            self.move()
            self.log_periodic("速度指令のBridge接続を待っています")
            return

        with self.lock:
            person = dict(self.person) if self.person is not None else None
            last_inference = self.last_inference
            last_seen = self.last_seen
            streak = self.detection_streak
            error = self.inference_error

        if error is not None:
            self.finish(f"YOLO推論エラー: {error}")
            return

        if (self.pose is None or self.front is None
                or self.left is None or self.right is None
                or last_inference is None
                or self.last_image is None
                or self.last_scan is None
                or self.last_odom is None
                or now - self.last_image > 4.0
                or now - self.last_scan > 3.0
                or now - self.last_odom > 3.0
                or now - last_inference > 2.5):
            self.move()
            self.log_periodic("センサー・YOLO推論の更新を待機中")
            return

        # 直進を命令する前に常に停止距離を確認。
        # SCAN/旋回は前進しないので、近い障害物があっても探索できる。
        if (self.state in ("APPROACH", "EXPLORE_MOVE",
                           "AVOID_SIDE", "AVOID_PASS")
                and self.front <= self.emergency_distance):
            self.finish(f"緊急停止: 前方={self.front:.2f}m")
            return

        if self.state == "SCAN":
            self.run_scan(person, streak, last_inference)
        elif self.state == "APPROACH":
            self.run_approach(person, last_seen, now)
        elif self.state.startswith("EXPLORE_"):
            self.run_explore()
        elif self.state.startswith("AVOID_"):
            self.run_avoidance()
        else:
            self.finish(f"不明な状態: {self.state}")


def main():
    rclpy.init()
    node = PersonNavigationController()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("Ctrl+Cにより停止します")
    finally:
        node.worker_stop.set()
        for _ in range(10):
            node.move()
            time.sleep(0.1)
        node.worker.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
