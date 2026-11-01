
#!/usr/bin/env python3
"""Gazebo Harmonic: 人物探索用の高難度障害物ワールドを生成する。

入力:  ~/pai_final/worlds/camera_world.sdf
出力:  ~/pai_final/worlds/object_world.sdf

既存の青いロボット・カメラ・LiDAR・基本障害物を保持し、
さらに箱・円柱障害物を追加する。人物は従来のActorを使用する。
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET


# (名前, 形状, x, y, 幅/半径, 奥行き, 高さ, 表示色 RGBA)
# 箱は幅・奥行き [m]、円柱は半径・奥行き=None。
# 青いロボットの初期位置 (0, 2) と人物 (10, 2) の周囲を空けている。
#
# medium専用：
# 障害物の数は6個のまま、追加4個を広い範囲に分散する。
# 既存のcentral_obstacle (2.7, 2.0)、
# left_obstacle (1.4, 3.5) は維持。
#
# 追加障害物の中心座標は、互いにおおむね3m以上離している。

MEDIUM_OBSTACLES = [
    (
        "hard_box_01", "box",
        4.8, -1.7, 0.85, 1.15, 1.5,
        "0.95 0.40 0.20 1"
    ),
    (
        "hard_box_02", "box",
        4.8, 5.8, 0.85, 1.20, 1.5,
        "0.95 0.40 0.20 1"
    ),
    (
        "hard_pillar_03", "cylinder",
        7.0, 1.6, 0.45, None, 1.6,
        "0.62 0.40 0.85 1"
    ),
    (
        "hard_box_04", "box",
        8.5, 4.5, 0.90, 1.40, 1.5,
        "0.95 0.40 0.20 1"
    ),
]


# hardの障害物配置は従来どおり。

OBSTACLES = [
    (
        "hard_box_01", "box",
        3.8, -0.9, 0.85, 1.15, 1.5,
        "0.95 0.40 0.20 1"
    ),
    (
        "hard_box_02", "box",
        3.8, 5.1, 0.85, 1.20, 1.5,
        "0.95 0.40 0.20 1"
    ),
    (
        "hard_pillar_03", "cylinder",
        5.0, 0.65, 0.50, None, 1.6,
        "0.62 0.40 0.85 1"
    ),
    (
        "hard_box_04", "box",
        5.2, 3.55, 0.90, 1.40, 1.5,
        "0.95 0.40 0.20 1"
    ),
    (
        "hard_box_05", "box",
        6.5, -1.15, 0.90, 1.15, 1.5,
        "0.95 0.40 0.20 1"
    ),
    (
        "hard_box_06", "box",
        6.6, 2.25, 1.00, 1.25, 1.5,
        "0.95 0.40 0.20 1"
    ),
    (
        "hard_pillar_07", "cylinder",
        6.7, 5.0, 0.55, None, 1.6,
        "0.62 0.40 0.85 1"
    ),
    (
        "hard_pillar_08", "cylinder",
        8.1, 0.4, 0.55, None, 1.6,
        "0.62 0.40 0.85 1"
    ),
    (
        "hard_box_09", "box",
        8.3, 4.05, 0.90, 1.15, 1.5,
        "0.95 0.40 0.20 1"
    ),
]


# ==========================================
# 障害物モデルを追加
# ==========================================

def add_obstacle(world, spec):
    name, shape, x, y, a, b, height, color = spec

    model = ET.SubElement(
        world,
        "model",
        name=name
    )

    ET.SubElement(model, "static").text = "true"

    ET.SubElement(model, "pose").text = (
        f"{x} {y} {height / 2:.3f} 0 0 0"
    )

    link = ET.SubElement(
        model,
        "link",
        name="link"
    )

    for kind in ("collision", "visual"):

        elem = ET.SubElement(
            link,
            kind,
            name=kind
        )

        geom = ET.SubElement(
            elem,
            "geometry"
        )

        shape_elem = ET.SubElement(
            geom,
            shape
        )

        if shape == "box":
            ET.SubElement(
                shape_elem,
                "size"
            ).text = f"{a} {b} {height}"

        else:
            ET.SubElement(
                shape_elem,
                "radius"
            ).text = str(a)

            ET.SubElement(
                shape_elem,
                "length"
            ).text = str(height)

        if kind == "visual":
            material = ET.SubElement(
                elem,
                "material"
            )

            ET.SubElement(
                material,
                "ambient"
            ).text = color

            ET.SubElement(
                material,
                "diffuse"
            ).text = color

    return model


# ==========================================
# 人物Actorを追加
# ==========================================

def add_actor(world, mesh, x, y, yaw):

    actor = ET.SubElement(
        world,
        "actor",
        name="target_person"
    )

    skin = ET.SubElement(actor, "skin")

    ET.SubElement(
        skin,
        "filename"
    ).text = mesh

    ET.SubElement(
        skin,
        "scale"
    ).text = "1.0"

    animation = ET.SubElement(
        actor,
        "animation",
        name="walk"
    )

    ET.SubElement(
        animation,
        "filename"
    ).text = mesh

    ET.SubElement(
        animation,
        "interpolate_x"
    ).text = "true"

    script = ET.SubElement(actor, "script")

    ET.SubElement(
        script,
        "loop"
    ).text = "true"

    ET.SubElement(
        script,
        "delay_start"
    ).text = "0"

    ET.SubElement(
        script,
        "auto_start"
    ).text = "true"

    trajectory = ET.SubElement(
        script,
        "trajectory",
        id="0",
        type="walk"
    )

    # 固定位置で歩行アニメーションを繰り返す
    for t in (0, 4):

        waypoint = ET.SubElement(
            trajectory,
            "waypoint"
        )

        ET.SubElement(
            waypoint,
            "time"
        ).text = str(t)

        ET.SubElement(
            waypoint,
            "pose"
        ).text = f"{x} {y} 1.0 0 0 {yaw}"

    return actor


# ==========================================
# メイン処理
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "既存のカメラ付きワールドから"
            "高難度の人物探索環境を作成"
        )
    )

    parser.add_argument(
        "--level",
        choices=("medium", "hard"),
        default="medium",
        help=(
            "medium=既存2個+追加4個、"
            "hard=既存2個+追加9個"
        )
    )

    parser.add_argument(
        "--x",
        type=float,
        default=10.0,
        help="人物のX座標"
    )

    parser.add_argument(
        "--y",
        type=float,
        default=2.0,
        help="人物のY座標"
    )

    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="人物の向き [rad]"
    )

    args = parser.parse_args()

    # --------------------------------------
    # ファイルパス
    # --------------------------------------

    root_dir = Path.home() / "pai_final" / "worlds"

    source = root_dir / "camera_world.sdf"
    output = root_dir / "object_world.sdf"

    if not source.is_file():
        raise FileNotFoundError(
            f"入力ワールドがありません: {source}"
        )

    # --------------------------------------
    # ワールド読み込み
    # --------------------------------------

    tree = ET.parse(source)

    world = tree.getroot().find("world")

    if world is None:
        raise RuntimeError(
            f"<world> が見つかりません: {source}"
        )

    # 必要なモデルの確認
    if world.find(
        "./model[@name='vehicle_blue']"
    ) is None:
        raise RuntimeError(
            "元のワールドに vehicle_blue がありません"
        )

    for name in (
        "central_obstacle",
        "left_obstacle"
    ):
        if world.find(
            f"./model[@name='{name}']"
        ) is None:
            raise RuntimeError(
                f"元のワールドに {name} がありません"
            )

    # --------------------------------------
    # 不要なモデルを除去
    # --------------------------------------

    for model in list(world.findall("model")):

        name = model.get("name", "")

        if (
            name in ("goal_marker", "target_chair")
            or name.startswith("hard_")
        ):
            world.remove(model)

    for actor in list(world.findall("actor")):

        if actor.get("name") == "target_person":
            world.remove(actor)

    # --------------------------------------
    # 障害物の追加
    # --------------------------------------

    if args.level == "medium":
        selected_obstacles = MEDIUM_OBSTACLES
    else:
        selected_obstacles = OBSTACLES

    for spec in selected_obstacles:
        add_obstacle(world, spec)

    # --------------------------------------
    # 人物メッシュの取得
    # --------------------------------------

    local_mesh = (
        Path.home()
        / ".gz/fuel/fuel.gazebosim.org"
        / "mingfei/models/actor/1/meshes/walk.dae"
    )

    if not local_mesh.is_file():
        raise FileNotFoundError(
            "人物メッシュがありません。"
            "前の人物実験で使用したwalk.daeを"
            f"確認してください: {local_mesh}"
        )

    # --------------------------------------
    # 人物を追加
    # --------------------------------------

    add_actor(
        world,
        str(local_mesh),
        args.x,
        args.y,
        args.yaw
    )

    # --------------------------------------
    # ワールドを保存
    # --------------------------------------

    ET.indent(tree, space="  ")

    tree.write(
        output,
        encoding="utf-8",
        xml_declaration=True
    )

    # 保存されたXMLが正常に読めるか確認
    ET.parse(output)

    # --------------------------------------
    # 実行結果を表示
    # --------------------------------------

    print(f"生成完了: {output}")
    print(f"難易度: {args.level}")

    print(
        f"追加障害物: {len(selected_obstacles)}個"
        f"（既存2個と合わせて計"
        f"{len(selected_obstacles) + 2}個）"
    )

    print(
        f"人物位置: ({args.x}, {args.y}), "
        f"yaw={args.yaw} rad"
    )


if __name__ == "__main__":
    main()
