# Assets and provenance

| Asset | Source | SHA-256 |
| --- | --- | --- |
| `models/yolov8n.pt` | Original Desktop project weight. Pretrained Ultralytics YOLOv8n person detection uses class 0 of the model's class map. [YOLOv8 model documentation](https://docs.ultralytics.com/models/yolov8/) | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `models/best.pt` | Member 1's cleaned YOLO11n checkpoint from `Campus-Threat-Detection-main.zip`, originally under `runs/detect/person1_clean/yolo11n_clean_v2/weights/`. [YOLO11 model documentation](https://docs.ultralytics.com/models/yolo11/) | `7c92caaf34cbd442d656f1fc0e78a05f79a2b7cc00cea1a533095e377117af6c` |
| `demo/approach-and-separate.mp4`, `demo/triggers.json`, `demo/no-trigger.json` | Generated for the project by `assets/make_demo.py`. The shapes, pair motion and two gate triggers are scripted. No real people or objects appear. | Regenerate with `python assets/make_demo.py`. |

The detector uses the five-class dataset published by Zbigniew Omiotek. The cleaned split and component metrics are described in `reference/detector/person1_clean_model_evaluation.md`. The extracted original detector source is under `src/`, and its retained contract tests run as part of the build suite. Cite the source dataset and its requested paper in the project submission. [Zenodo dataset](https://zenodo.org/records/16422779) · [Dataset paper](https://doi.org/10.3390/electronics14173540)

ByteTrack is the association method implemented by the Trackers package. The project contributes its gate, pipeline, analysis and evaluation. [ByteTrack paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf) · [Trackers repository](https://github.com/roboflow/trackers)

The included checkpoints are local copies of the supplied assets. Keep upstream notices and check the applicable team and Ultralytics terms before redistributing weights or derivative code. [Ultralytics license information](https://www.ultralytics.com/license)
