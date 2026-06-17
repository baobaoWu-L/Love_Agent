"""
人像识别模块：人脸检测、人脸识别、人脸注册

为 Agent 提供视觉身份识别能力，支持基于记忆库的特征向量匹配。
所有可选依赖均使用懒加载，缺失时以降级模式运行。
"""
import logging
import time
from pathlib import Path
from typing import Optional

from config.agent_config import MEMORY_DIR

logger = logging.getLogger("loveflow.FACE_RECOGNITION")

# 尝试导入 face_recognition
face_recognition = None
try:
    import face_recognition as _fr

    face_recognition = _fr
except ImportError:
    logger.warning("face_recognition 未安装，尝试导入 DeepFace 作为备选...")

# 尝试导入 DeepFace（备选方案）
DeepFace = None
try:
    from deepface import DeepFace as _df

    DeepFace = _df
    logger.info("已加载 DeepFace 作为人像识别引擎")
except ImportError:
    if face_recognition is None:
        logger.warning(
            "face_recognition 和 deepface 均未安装，人像识别模块将以降级模式运行。"
            "请执行: pip install face_recognition 或 pip install deepface"
        )

# 尝试导入 PIL（用于图片预处理）
Image = None
try:
    from PIL import Image as _PIL

    Image = _PIL
except ImportError:
    pass

# 记忆库中的人脸存储目录
FACE_VECTORS_DIR = MEMORY_DIR / "face_vectors"
FACE_VECTORS_DIR.mkdir(parents=True, exist_ok=True)


class FaceRecognizer:
    """人像识别器：检测、识别、注册人脸"""

    def __init__(self):
        """初始化人像识别器"""
        self._engine = None
        if face_recognition is not None:
            self._engine = "face_recognition"
            logger.info("人像识别引擎: face_recognition")
        elif DeepFace is not None:
            self._engine = "deepface"
            logger.info("人像识别引擎: DeepFace")
        else:
            logger.warning("无人像识别引擎可用，功能将受限")

    # ==================== 人脸检测 ====================

    def detect_faces(self, image_path: str) -> list[dict]:
        """
        检测图片中的人脸位置

        Args:
            image_path: 图片文件路径

        Returns:
            list[dict]: 人脸列表，每项包含位置信息
                [
                    {
                        "top": int, "right": int, "bottom": int, "left": int,
                        "confidence": float (可选),
                    }
                ]
        """
        if self._engine is None:
            logger.warning("无人像识别引擎可用，无法检测人脸")
            return []

        img_path = Path(image_path)
        if not img_path.exists():
            logger.error(f"图片文件不存在: {image_path}")
            return []

        try:
            if self._engine == "face_recognition":
                return self._detect_with_fr(img_path)
            elif self._engine == "deepface":
                return self._detect_with_deepface(img_path)
        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            return []

    def _detect_with_fr(self, img_path: Path) -> list[dict]:
        """使用 face_recognition 库检测人脸"""
        image = face_recognition.load_image_file(str(img_path))
        locations = face_recognition.face_locations(image)

        results = []
        for loc in locations:
            top, right, bottom, left = loc
            results.append({
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
                "left": int(left),
            })

        logger.info(f"检测到 {len(results)} 张人脸 (引擎: face_recognition)")
        return results

    def _detect_with_deepface(self, img_path: Path) -> list[dict]:
        """使用 DeepFace 检测人脸"""
        try:
            result = DeepFace.extract_faces(
                img_path=str(img_path),
                detector_backend="opencv",
                enforce_detection=False,
            )

            faces = []
            for face_data in result:
                area = face_data.get("area", {})
                confidence = face_data.get("confidence", 0)
                faces.append({
                    "top": int(area.get("y", 0)),
                    "right": int(area.get("x", 0) + area.get("w", 0)),
                    "bottom": int(area.get("y", 0) + area.get("h", 0)),
                    "left": int(area.get("x", 0)),
                    "confidence": float(confidence),
                })

            logger.info(f"检测到 {len(faces)} 张人脸 (引擎: DeepFace)")
            return faces
        except Exception as e:
            logger.error(f"DeepFace 检测失败: {e}")
            return []

    # ==================== 人脸识别 ====================

    def recognize(self, image_path: str) -> list[dict]:
        """
        识别图片中的人脸，与记忆库中人脸特征向量匹配

        Args:
            image_path: 图片文件路径

        Returns:
            list[dict]: 识别结果列表
                [
                    {
                        "name": str,         # 匹配到的姓名
                        "confidence": float,  # 匹配置信度 (0~1)
                        "location": dict,     # 人脸位置
                        "memory_id": int,     # 记忆库 ID
                    }
                ]
        """
        if self._engine is None:
            logger.warning("无人像识别引擎可用，无法识别人脸")
            return []

        img_path = Path(image_path)
        if not img_path.exists():
            logger.error(f"图片文件不存在: {image_path}")
            return []

        try:
            if self._engine == "face_recognition":
                return self._recognize_with_fr(img_path)
            elif self._engine == "deepface":
                return self._recognize_with_deepface(img_path)
        except Exception as e:
            logger.error(f"人脸识别失败: {e}")
            return []

    def _recognize_with_fr(self, img_path: Path) -> list[dict]:
        """使用 face_recognition 进行识别"""
        # 从记忆库中加载所有人脸特征
        known_encodings, known_names, known_ids = self._load_face_vectors()

        if not known_encodings:
            logger.info("记忆库中无人脸数据，无法进行识别")
            return []

        # 检测并编码当前图片中的人脸
        image = face_recognition.load_image_file(str(img_path))
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        results = []
        for i, encoding in enumerate(encodings):
            # 与记忆库中所有人脸比较
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_idx = distances.argmin() if len(distances) > 0 else -1

            loc = locations[i]
            if best_match_idx >= 0:
                confidence = 1 - float(distances[best_match_idx])
                results.append({
                    "name": known_names[best_match_idx],
                    "confidence": round(confidence, 4),
                    "location": {
                        "top": int(loc[0]), "right": int(loc[1]),
                        "bottom": int(loc[2]), "left": int(loc[3]),
                    },
                    "memory_id": known_ids[best_match_idx],
                })
            else:
                results.append({
                    "name": "未知",
                    "confidence": 0.0,
                    "location": {
                        "top": int(loc[0]), "right": int(loc[1]),
                        "bottom": int(loc[2]), "left": int(loc[3]),
                    },
                    "memory_id": -1,
                })

        return results

    def _recognize_with_deepface(self, img_path: Path) -> list[dict]:
        """使用 DeepFace 进行识别"""
        known_names, known_ids = self._load_face_metadata()
        if not known_names:
            logger.info("记忆库中无人脸数据")
            return []

        results = []
        for name, mid in zip(known_names, known_ids):
            try:
                # 查找对应的人脸图片目录
                person_dir = FACE_VECTORS_DIR / str(mid)
                if not person_dir.exists():
                    continue

                ref_images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
                if not ref_images:
                    continue

                # 用第一张参考图片做验证
                result = DeepFace.verify(
                    img1_path=str(img_path),
                    img2_path=str(ref_images[0]),
                    enforce_detection=False,
                )

                if result.get("verified"):
                    results.append({
                        "name": name,
                        "confidence": round(1 - result.get("distance", 1), 4),
                        "location": {},
                        "memory_id": mid,
                    })
            except Exception as e:
                logger.debug(f"DeepFace 比对 {name} 失败: {e}")

        return results

    # ==================== 人脸注册 ====================

    def register_face(self, image_path: str, name: str) -> int:
        """
        注册人脸到记忆库

        提取人脸特征向量，保存到 SQLite 记忆库（doc_type='face'），
        同时将参考图片保存到本地文件系统。

        Args:
            image_path: 包含人脸的图片文件路径
            name: 人名

        Returns:
            int: 记忆库中的记录 ID，失败返回 -1
        """
        if self._engine is None:
            logger.warning("无人像识别引擎可用，无法注册人脸")
            return -1

        img_path = Path(image_path)
        if not img_path.exists():
            logger.error(f"图片文件不存在: {image_path}")
            return -1

        try:
            if self._engine == "face_recognition":
                memory_id = self._register_with_fr(img_path, name)
            else:
                memory_id = self._register_with_deepface(img_path, name)

            if memory_id > 0:
                logger.info(f"人脸注册成功: {name} (ID: {memory_id})")
            return memory_id
        except Exception as e:
            logger.error(f"人脸注册失败: {e}")
            return -1

    def _register_with_fr(self, img_path: Path, name: str) -> int:
        """使用 face_recognition 提取特征并注册"""
        from agent.core.memory import add_memory

        image = face_recognition.load_image_file(str(img_path))
        locations = face_recognition.face_locations(image)

        if not locations:
            logger.warning(f"图片中未检测到人脸: {img_path}")
            return -1

        encodings = face_recognition.face_encodings(image, locations)

        if not encodings:
            logger.warning(f"无法提取人脸特征向量: {img_path}")
            return -1

        # 取第一张人脸的特征向量
        vector = list(encodings[0])

        # 保存参考图片
        mid = add_memory(
            content=f"人脸注册: {name}",
            doc_type="face",
            title=name,
            tags=["人脸", name],
            source=str(img_path),
            importance=3,
            vector=vector,
        )

        # 复制参考图片到人脸目录
        person_dir = FACE_VECTORS_DIR / str(mid)
        person_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(img_path), str(person_dir / img_path.name))

        return mid

    def _register_with_deepface(self, img_path: Path, name: str) -> int:
        """使用 DeepFace 提取特征并注册"""
        from agent.core.memory import add_memory

        try:
            # DeepFace 的 represent 方法返回嵌入向量
            embedding_data = DeepFace.represent(
                img_path=str(img_path),
                model_name="Facenet",
                enforce_detection=False,
            )

            if not embedding_data:
                logger.warning(f"无法提取人脸特征: {img_path}")
                return -1

            # 取第一个检测到的面部
            vector = embedding_data[0].get("embedding", [])
            if not vector:
                logger.warning("提取的人脸特征向量为空")
                return -1

            mid = add_memory(
                content=f"人脸注册: {name}",
                doc_type="face",
                title=name,
                tags=["人脸", name],
                source=str(img_path),
                importance=3,
                vector=vector,
            )

            # 复制参考图片
            person_dir = FACE_VECTORS_DIR / str(mid)
            person_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(img_path), str(person_dir / img_path.name))

            return mid
        except Exception as e:
            logger.error(f"DeepFace 注册失败: {e}")
            return -1

    # ==================== 查询记忆 ====================

    def find_known_faces(self, name: Optional[str] = None) -> list[dict]:
        """
        查询已注册的人脸记忆

        Args:
            name: 按姓名筛选（可选）

        Returns:
            list[dict]: 人脸记忆列表
        """
        from agent.core.memory import search_memory, list_memories

        try:
            if name:
                # 按姓名搜索
                results = search_memory(name, doc_type="face", limit=20)
            else:
                # 列出所有人脸
                results = list_memories(doc_type="face", limit=100)

            # 清理向量数据（体积大且外部不需要）
            for r in results:
                r.pop("vector", None)

            return results
        except Exception as e:
            logger.error(f"查询人脸记忆失败: {e}")
            return []

    # ==================== 内部辅助 ====================

    def _load_face_vectors(self) -> tuple[list, list[str], list[int]]:
        """
        从记忆库加载所有已注册的人脸特征向量

        Returns:
            (encodings, names, ids): 特征向量列表、姓名列表、记忆 ID 列表
        """
        from agent.core.memory import search_memory

        try:
            memories = search_memory("", doc_type="face", limit=500)
        except Exception:
            memories = []

        encodings = []
        names = []
        ids = []

        for mem in memories:
            vec = mem.get("vector")
            if vec and len(vec) == 128:  # face_recognition 使用 128 维向量
                import numpy as np

                encodings.append(np.array(vec))
                names.append(mem.get("title", "未知"))
                ids.append(mem["id"])

        return encodings, names, ids

    def _load_face_metadata(self) -> tuple[list[str], list[int]]:
        """加载人脸元数据（姓名和 ID），不含特征向量"""
        from agent.core.memory import list_memories

        try:
            memories = list_memories(doc_type="face", limit=500)
        except Exception:
            memories = []

        names = [m.get("title", "未知") for m in memories]
        ids = [m["id"] for m in memories]
        return names, ids

    # ==================== 上下文注入 ====================

    def get_context(self) -> Optional[str]:
        """
        返回人像识别能力摘要，供 Agent 上下文注入

        Returns:
            Optional[str]: 人脸识别状态文本，无引擎时返回 None
        """
        if self._engine is None:
            return None

        try:
            known = self.find_known_faces()
            parts = ["【人像识别】"]

            if self._engine == "face_recognition":
                parts.append("引擎: face_recognition")
            else:
                parts.append("引擎: DeepFace")

            parts.append(f"已注册人脸: {len(known)} 人")

            if known:
                names = [k.get("title", "未知") for k in known[:5]]
                parts.append(f"已知人物: {', '.join(names)}")
                if len(known) > 5:
                    parts.append(f"...等 {len(known)} 人")

            return " | ".join(parts)
        except Exception as e:
            logger.error(f"生成人像识别上下文失败: {e}")
            return None
